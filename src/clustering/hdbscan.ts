import type { BaseClustering, DataMatrix, HDBSCANParams } from './types';
import type { ClusterRepresentations } from './representations';
import * as tf from '../backend/adapter';
import { is_tensor } from '../tensor/tensor_guards';
import { kdistance } from '../distance/kdistance';
import { mutual_reachability } from '../graph/mutual_reachability';
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
 * Density and tree primitives are consumed from their domain modules rather
 * than reimplemented: k-distance from `distance/kdistance`, mutual reachability
 * and the minimum spanning tree from `graph/`, and the condensed tree from
 * `graph/condensation_tree`.
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
   * Builds the dense distance matrix the density pipeline operates on.
   *
   * Native metrics are computed in plain JavaScript float64 rather than via
   * the TensorFlow.js backend: the rest of the pipeline (MST, condensed tree)
   * is float64, and a float32 round-trip would perturb mutual-reachability
   * weights enough to break exact parity with scikit-learn on tie-free data.
   */
  private async distance_matrix(X: DataMatrix): Promise<number[][]> {
    const metric = this.params.metric ?? 'euclidean';
    const rows: number[][] = is_tensor(X)
      ? ((await (X as tf.Tensor2D).array()) as number[][])
      : (X as number[][]);

    if (metric === 'precomputed') {
      const n = rows.length;
      for (const row of rows) {
        if (row.length !== n) {
          throw new Error(
            'precomputed metric requires a square (n, n) distance matrix.',
          );
        }
      }
      return rows;
    }

    const n = rows.length;
    const d = n > 0 ? rows[0].length : 0;
    for (const row of rows) {
      if (row.length !== d) {
        throw new Error(
          'Input data must be rectangular: every sample needs the same feature count.',
        );
      }
    }
    const D: number[][] = Array.from({ length: n }, () =>
      new Array<number>(n).fill(0),
    );
    for (let i = 0; i < n; i++) {
      const ri = rows[i];
      for (let j = i + 1; j < n; j++) {
        const rj = rows[j];
        let s = 0;
        if (metric === 'manhattan') {
          for (let f = 0; f < ri.length; f++) s += Math.abs(ri[f] - rj[f]);
        } else {
          for (let f = 0; f < ri.length; f++) {
            const diff = ri[f] - rj[f];
            s += diff * diff;
          }
          s = Math.sqrt(s);
        }
        D[i][j] = s;
        D[j][i] = s;
      }
    }
    return D;
  }

  async fit(X: DataMatrix): Promise<void> {
    const D = await this.distance_matrix(X);
    const n = D.length;

    if (n === 0) {
      throw new Error('Input data must contain at least one sample.');
    }

    // All input validation must pass before dispose() so a failed re-fit
    // leaves the prior fitted state intact (distance_matrix validates shape;
    // the n===0 guard above covers empty input).
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

    // Core distances: k-th nearest neighbour distance (self counts first).
    const neighbor_distances = D.map((row) => [...row].sort((a, b) => a - b));
    const core = kdistance(neighbor_distances, min_samples);

    // Mutual-reachability graph -> minimum spanning tree -> condensed tree.
    const mreach = mutual_reachability(D, core);
    const mst = minimum_spanning_tree(mreach);
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
  }

  async fit_predict(X: DataMatrix): Promise<number[]> {
    await this.fit(X);
    if (this.labels_ == null) {
      throw new Error('HDBSCAN.fit did not compute labels.');
    }
    return this.labels_;
  }
}
