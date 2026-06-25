import type { BaseClustering, DataMatrix, HDBSCANParams } from './types';
import type { ClusterRepresentations } from './representations';
import * as tf from '../backend/adapter';
import { is_tensor } from '../tensor/tensor_guards';
import { pairwise_distance_matrix } from '../distance/pairwise_distance';
import { minimum_spanning_tree } from '../graph/minimum_spanning_tree';
import {
  build_condensation_tree,
  excess_of_mass,
  extract_labels,
} from '../graph/condensation_tree';

/**
 * HDBSCAN — hierarchical density-based clustering.
 *
 * Discovers clusters of varying density without a preset cluster count and
 * flags samples in sparse regions as noise (`-1`). The estimator builds the
 * mutual-reachability graph from per-point core distances, takes its minimum
 * spanning tree, condenses the resulting single-linkage hierarchy, and selects
 * stable clusters by Excess of Mass. It is fit-only — like
 * AgglomerativeClustering there is no principled `predict` for unseen points.
 *
 * The front-half (distance matrix, core distances, mutual reachability) runs
 * on the TensorFlow.js backend in a single fused `tf.tidy` inside `fit`. Core
 * distances use a `tf.topk` order-statistic; mutual reachability is a broadcast
 * `tf.maximum`. The minimum spanning tree comes from `graph/minimum_spanning_tree`
 * and the condensed tree from `graph/condensation_tree`.
 *
 * Parity: labels and probabilities match scikit-learn closely but not
 * bit-for-bit. Mutual-reachability weight ties are ordered differently across
 * implementations (numpy's unstable `argsort` over the MST edges), which shifts
 * a few boundary points. The condensed-tree + Excess-of-Mass core is itself
 * exact — it reproduces scikit-learn's labels and probabilities verbatim when
 * fed scikit-learn's own single-linkage hierarchy (see
 * `condensation_tree.test.ts`).
 */
export class HDBSCAN
  implements BaseClustering<HDBSCANParams>, ClusterRepresentations
{
  public readonly params: HDBSCANParams;

  /** Cluster labels: integers >= 0, `-1` for noise. Null until `fit`. */
  public labels_: number[] | null = null;

  /** Per-sample membership strength in `[0, 1]`. Null until `fit`. */
  public probabilities_: number[] | null = null;

  /**
   * Most-persistent exemplar sample index per 0-based cluster label.
   *
   * Library-defined: scikit-learn's HDBSCAN has no equivalent attribute. The
   * exemplar of a cluster is the point that persists to the highest λ in the
   * condensed tree; ties resolve towards the lowest sample index. Populated
   * only when `store_exemplars` is set; null otherwise.
   */
  public exemplar_indices_: Map<number, number> | null = null;

  private static readonly DEFAULT_MIN_CLUSTER_SIZE = 5;

  constructor(params: Partial<HDBSCANParams> = {}) {
    this.params = { ...params };
    HDBSCAN.validate_params(this.params);
  }

  private static validate_params(params: HDBSCANParams): void {
    const { min_cluster_size, min_samples, metric, cluster_selection_method } =
      params;

    if (
      min_cluster_size !== undefined &&
      (!Number.isInteger(min_cluster_size) || min_cluster_size < 2)
    ) {
      throw new Error('min_cluster_size must be an integer >= 2 when given.');
    }
    if (
      min_samples !== undefined &&
      (!Number.isInteger(min_samples) || min_samples < 1)
    ) {
      throw new Error('min_samples must be an integer >= 1 when given.');
    }
    if (
      metric !== undefined &&
      metric !== 'euclidean' &&
      metric !== 'manhattan' &&
      metric !== 'precomputed'
    ) {
      throw new Error(
        "metric must be 'euclidean', 'manhattan', or 'precomputed'.",
      );
    }
    if (
      cluster_selection_method !== undefined &&
      cluster_selection_method !== 'eom' &&
      cluster_selection_method !== 'leaf'
    ) {
      throw new Error("cluster_selection_method must be 'eom' or 'leaf'.");
    }
  }

  /** Resets fitted state. HDBSCAN keeps no tensors as instance state. */
  public dispose(): void {
    this.labels_ = null;
    this.probabilities_ = null;
    this.exemplar_indices_ = null;
  }

  /**
   * Builds the dense pairwise distance matrix as an `(n, n)` `Tensor2D`.
   *
   * Native `euclidean`/`manhattan` metrics are delegated to
   * `pairwise_distance_matrix`, which runs on the TensorFlow.js backend in
   * float32. A `precomputed` matrix is validated for squareness and uploaded
   * to a `Tensor2D` unchanged. The result is the first front-half stage; `fit`
   * consumes it for the core-distance and mutual-reachability computation.
   *
   * The returned tensor is always freshly owned: `fit` disposes it without
   * touching a caller-supplied tensor (the precomputed-tensor case is cloned).
   */
  private distance_matrix(X: DataMatrix): tf.Tensor2D {
    const metric = this.params.metric ?? 'euclidean';

    // Reject empty input before any tensor allocation (tf.tensor2d cannot
    // infer a shape from `[]`) and before fit() reaches dispose(), so a failed
    // re-fit leaves prior fitted state intact.
    const n = is_tensor(X)
      ? (X as tf.Tensor2D).shape[0]
      : (X as number[][]).length;
    if (n === 0) {
      throw new Error('Input data must contain at least one sample.');
    }

    if (metric === 'precomputed') {
      if (is_tensor(X)) {
        const matrix = X as tf.Tensor2D;
        if (matrix.shape[0] !== matrix.shape[1]) {
          throw new Error(
            'precomputed metric requires a square (n, n) distance matrix.',
          );
        }
        return matrix.clone();
      }
      const rows = X as number[][];
      for (const row of rows) {
        if (row.length !== n) {
          throw new Error(
            'precomputed metric requires a square (n, n) distance matrix.',
          );
        }
      }
      return tf.tensor2d(rows);
    }

    if (is_tensor(X)) {
      return pairwise_distance_matrix(X as tf.Tensor2D, metric);
    }

    const rows = X as number[][];
    const d = rows[0].length;
    for (const row of rows) {
      if (row.length !== d) {
        throw new Error(
          'Input data must be rectangular: every sample needs the same feature count.',
        );
      }
    }
    // For the euclidean metric, distances come from pairwise_euclidean_matrix,
    // which uses the gram identity ‖x‖²+‖y‖²−2·xᵀy. Cancellation can yield
    // tiny negative squared distances; the float32 `maximum(·, 0)` clamp inside
    // the helper pins them to zero. Labels are verified robust to the resulting
    // float32 drift — hdbscan.test.ts matches the scikit-learn oracle exactly
    // under the task-54.2 tolerances.
    const points = tf.tensor2d(rows);
    try {
      return pairwise_distance_matrix(points, metric);
    } finally {
      points.dispose();
    }
  }

  /**
   * Computes per-point core distances using `tf.topk` on the negated distance
   * matrix. The core distance of point i is the distance to its
   * (min_samples − 1)-th nearest neighbour, counting self as neighbour 0.
   *
   * Negation trick: `tf.topk` returns the largest values first, so negating
   * `D_tensor` makes the min_samples smallest distances rank first. Column
   * (min_samples − 1) of the negated result is the negated k-th order
   * statistic; negating again recovers the core distance. The diagonal (self,
   * distance 0) is always the least-negative value and occupies index 0.
   *
   * The returned tensor is owned by the caller and must be disposed after use.
   */
  private core_distances(
    D_tensor: tf.Tensor2D,
    min_samples: number,
  ): tf.Tensor1D {
    return tf.tidy(() => {
      // Negate so topk (largest-first) selects the min_samples smallest distances.
      const { values } = tf.topk(D_tensor.neg(), min_samples);
      // Column min_samples-1 holds the negated k-th order statistic; un-negate.
      return values
        .slice([0, min_samples - 1], [-1, 1])
        .reshape([-1])
        .neg();
    }) as tf.Tensor1D; // tf.tidy widens to Tensor<Rank>; body is always rank-1
  }

  async fit(X: DataMatrix): Promise<void> {
    // distance_matrix validates input shape and rejects empty input, all
    // before dispose() so a failed re-fit leaves prior fitted state intact. It
    // returns an (n, n) Tensor2D this method owns and disposes exactly once.
    const D_tensor = this.distance_matrix(X);
    let M_tensor: tf.Tensor2D | null = null;
    try {
      const n = D_tensor.shape[0];

      this.dispose();
      // Intentional deviation from scikit-learn (which raises for n_samples=1):
      // a lone sample is trivially noise, so degrade gracefully.
      if (n === 1) {
        this.labels_ = [-1];
        this.probabilities_ = [0];
        this.exemplar_indices_ = this.params.store_exemplars ? new Map() : null;
        return;
      }

      const min_cluster_size =
        this.params.min_cluster_size ?? HDBSCAN.DEFAULT_MIN_CLUSTER_SIZE;
      // min_samples defaults to min_cluster_size, clamped to the sample count.
      // Intentional deviation from scikit-learn, which raises when
      // min_samples > n_samples; the clamp keeps small inputs usable.
      const min_samples = Math.min(
        this.params.min_samples ?? min_cluster_size,
        n,
      );

      // Fuse core distances and mutual-reachability on-tensor in a single
      // tf.tidy; the core vector, its two reshaped views, and the intermediate
      // tf.maximum are freed on exit; M_tensor is the sole output.
      // M[i,j] = max(core[i], core[j], D[i,j]) via broadcast tf.maximum.
      M_tensor = tf.tidy(() => {
        const core = this.core_distances(D_tensor, min_samples);
        return tf.maximum(
          tf.maximum(core.reshape([n, 1]), core.reshape([1, n])),
          D_tensor,
        );
      }) as tf.Tensor2D;

      // Single GPU→CPU readback: flat row-major Float32Array of length n*n.
      const mreach_flat = (await M_tensor.data()) as Float32Array;
      M_tensor.dispose();
      M_tensor = null;

      const mst = minimum_spanning_tree(mreach_flat, n);
      const tree = build_condensation_tree(mst, n, min_cluster_size);

      const selected = excess_of_mass(tree, n, {
        cluster_selection_method: this.params.cluster_selection_method ?? 'eom',
        cluster_selection_epsilon: this.params.cluster_selection_epsilon ?? 0,
      });

      const { labels, probabilities, exemplar_indices } = extract_labels(
        tree,
        selected,
        n,
      );

      this.labels_ = labels;
      this.probabilities_ = probabilities;
      this.exemplar_indices_ = this.params.store_exemplars
        ? exemplar_indices
        : null;
    } finally {
      M_tensor?.dispose();
      D_tensor.dispose();
    }
  }

  async fit_predict(X: DataMatrix): Promise<number[]> {
    await this.fit(X);
    if (this.labels_ == null) {
      throw new Error('HDBSCAN.fit did not compute labels.');
    }
    return this.labels_;
  }
}
