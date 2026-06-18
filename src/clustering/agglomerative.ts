import type {
  DataMatrix,
  AgglomerativeClusteringParams,
  BaseClustering,
} from './types';
import * as tf from '../backend/adapter';
import { is_tensor } from '../tensor/tensor_guards';
import { MergeRecord, nn_chain_cluster } from './linkage';
import type { ClusterRepresentations } from './representations';
import { select_medoids } from './medoid_selection';

/**
 * Agglomerative (hierarchical) clustering using nearest-neighbor chain merges
 * with Lance–Williams distance updates.
 *
 * Builds the full reducible-linkage tree in O(n²) time for single, complete,
 * average, and Ward linkage, then cuts that tree by `n_clusters` or
 * `distance_threshold`.
 *
 * Distances are computed in float64 with the same definitions as
 * scikit-learn's `pairwise_distances`, and the Ward update is arranged to round
 * bit-identically to scipy. Ward, complete, and average linkage therefore
 * reproduce scikit-learn's partitions exactly, including the resolution of
 * exactly-tied merge distances on degenerate data such as integer grids or
 * duplicate points.
 *
 * Single linkage produces a correct single-linkage tree, but its resolution of
 * tied distances can differ from scikit-learn's: single linkage admits several
 * equally-valid merge orders, and scikit-learn, scipy, and the nearest-neighbor
 * chain each break exact ties differently. Partitions agree whenever the data
 * contains no exactly-tied distances (the usual case for continuous inputs).
 */
export class AgglomerativeClustering
  implements
    BaseClustering<AgglomerativeClusteringParams>,
    ClusterRepresentations
{
  public readonly params: AgglomerativeClusteringParams;

  /**
   * Index of the representative sample (medoid) per cluster, populated by
   * {@link compute_medoids}. Position `c` holds cluster `c`'s medoid index, or
   * `-1` if that cluster has no assigned samples.
   */
  public medoid_indices_: Int32Array | null = null;

  /**
   * Cluster labels produced by `fit` / `fit_predict`.
   *
   * Populated after calling `fit`.
   */
  public labels_: number[] | null = null;

  /**
   * Children recorded for each merge performed during agglomeration.
   * Shape: `(n_merges, 2)` where `n_merges = n_samples - n_clusters` (merging
   * stops once `n_clusters` clusters remain, so the full tree is not built), or
   * the number of merges strictly below `distance_threshold` when that
   * stopping criterion is used. Each row gives the global ids of the two merged
   * clusters (sklearn convention: original samples are `0..n-1`, each merge
   * creates id `n, n+1, ...`). Populated by `fit`.
   */
  public children_: number[][] | null = null;

  /**
   * Distance at which each merge in `children_` occurred. Aligned 1:1 with
   * `children_` (same length and order). Enables dendrogram cutting and
   * inspection of merge heights. Populated by `fit`.
   */
  public distances_: number[] | null = null;

  /**
   * Number of leaves in the hierarchical clustering tree, equal to the number
   * of input samples. Populated by `fit`.
   */
  public n_leaves_: number | null = null;

  private static readonly VALID_LINKAGES = [
    'ward',
    'complete',
    'average',
    'single',
  ] as const;

  private static readonly VALID_METRICS = [
    'euclidean',
    'manhattan',
    'cosine',
    'precomputed',
  ] as const;

  /**
   * @param params - Configuration for agglomerative clustering.
   */
  constructor(params: AgglomerativeClusteringParams) {
    this.params = { ...params };
    AgglomerativeClustering.validate_params(this.params);
  }

  /**
   * Fits the agglomerative clustering model to the input data.
   *
   * @param _X - Input data matrix of shape [n_samples, n_features].
   * @returns A promise that resolves when fitting is complete.
   * @throws {Error} If input is empty or n_clusters exceeds n_samples.
   *
   * @example
   * ```typescript
   * const agg = new AgglomerativeClustering({ n_clusters: 3 });
   * await agg.fit([[1, 2], [3, 4], [5, 6]]);
   * console.log(agg.labels_);
   * ```
   */
  async fit(_X: DataMatrix): Promise<void> {
    const { metric = 'euclidean', linkage = 'ward' } = this.params;
    const use_threshold = this.params.distance_threshold != null;

    // Resolve a flat row-major (i*n+j) Float64 distance matrix `D` and
    // `n_samples`, either from a precomputed matrix or by computing pairwise
    // distances from the data points.
    let D: Float64Array;
    let n_samples: number;

    if (metric === 'precomputed') {
      const raw = is_tensor(_X) ? await (_X as tf.Tensor2D).array() : _X;
      AgglomerativeClustering.validate_precomputed(raw);
      n_samples = raw.length;
      D = new Float64Array(n_samples * n_samples);
      for (let i = 0; i < n_samples; i++) {
        const row = raw[i];
        for (let j = 0; j < n_samples; j++) {
          D[i * n_samples + j] = row[j];
        }
      }
    } else {
      // Materialize the coordinates as a float64 array and compute pairwise
      // distances directly. scikit-learn computes distances in float64;
      // matching that precision (instead of the tfjs float32 backend) is what
      // lets ward/complete/average ties resolve identically to sklearn on
      // degenerate data such as integer grids or duplicate points.
      const data: number[][] = is_tensor(_X)
        ? ((await (_X as tf.Tensor2D).array()) as number[][])
        : (_X as number[][]);

      n_samples = data.length;

      if (n_samples === 0) {
        throw new Error('Input X must contain at least one sample.');
      }

      D = AgglomerativeClustering.compute_distance_matrix(data, metric);
    }

    if (!use_threshold && this.params.n_clusters! > n_samples) {
      throw new Error('n_clusters cannot exceed number of samples.');
    }

    // Handle trivial case of single sample separately
    if (n_samples === 1) {
      this.labels_ = [0];
      this.children_ = [];
      this.distances_ = [];
      this.n_leaves_ = 1;
      return;
    }

    // NN-chain is exact for the reducible linkages supported here: single,
    // complete, average, and Ward. It must build the full tree before cutting.
    const all_merges = nn_chain_cluster(D, n_samples, linkage);
    const merges = use_threshold
      ? all_merges.filter((m) => m.distance < this.params.distance_threshold!)
      : all_merges.slice(0, n_samples - this.params.n_clusters!);

    const children = AgglomerativeClustering.build_children(
      merges,
      n_samples,
    );

    // Derive flat cluster labels using Union-Find over the merge history
    const parent = new Int32Array(n_samples);
    for (let i = 0; i < n_samples; i++) parent[i] = i;

    function find(x: number): number {
      while (parent[x] !== x) {
        parent[x] = parent[parent[x]]; // path compression
        x = parent[x];
      }
      return x;
    }

    // Replay all merges — the survivor absorbs the removed cluster
    for (const m of merges) {
      const ra = find(m.cluster_a);
      const rb = find(m.cluster_b);
      parent[rb] = ra;
    }

    // Assign contiguous labels 0..n_clusters-1
    const labels = new Array<number>(n_samples);
    const root_to_label = new Map<number, number>();
    let next_label = 0;
    for (let i = 0; i < n_samples; i++) {
      const root = find(i);
      if (!root_to_label.has(root)) {
        root_to_label.set(root, next_label++);
      }
      labels[i] = root_to_label.get(root)!;
    }

    this.labels_ = labels;
    this.children_ = children;
    this.distances_ = merges.map((m) => m.distance);
    this.n_leaves_ = n_samples;
  }

  /**
   * Fits the model and returns cluster labels.
   *
   * @param _X - Input data matrix of shape [n_samples, n_features].
   * @returns Array of cluster labels for each sample.
   * @throws {Error} If input is empty or n_clusters exceeds n_samples.
   */
  async fit_predict(_X: DataMatrix): Promise<number[]> {
    await this.fit(_X);
    if (this.labels_ == null) {
      throw new Error('AgglomerativeClustering failed to compute labels.');
    }
    return this.labels_;
  }

  /**
   * Computes the representative sample (medoid) of every cluster — the sample
   * closest to its cluster mean under the estimator's metric — and stores them
   * in {@link medoid_indices_}.
   *
   * @param X The data the model was fitted on (same row order as `labels_`).
   * @returns The populated `medoid_indices_`.
   * @throws {Error} If called before `fit()`.
   */
  async compute_medoids(X: DataMatrix): Promise<Int32Array> {
    if (this.labels_ == null) {
      throw new Error(
        'AgglomerativeClustering.compute_medoids called before fit().',
      );
    }
    const metric = this.params.metric ?? 'euclidean';
    if (metric === 'precomputed') {
      throw new Error(
        'AgglomerativeClustering.compute_medoids is not supported with ' +
          'metric "precomputed": medoid selection requires feature vectors, ' +
          'not a precomputed distance matrix.',
      );
    }
    // The cluster count is determined by the fitted partition: with
    // `distance_threshold` (rather than `n_clusters`) the number of clusters is
    // data-driven, so derive it from the contiguous `0..k-1` labels.
    let n_clusters = 0;
    for (const label of this.labels_) {
      if (label >= n_clusters) n_clusters = label + 1;
    }
    const { indices } = await select_medoids(
      X,
      this.labels_,
      n_clusters,
      metric,
    );
    this.medoid_indices_ = indices;
    return indices;
  }

  private static validate_params(params: AgglomerativeClusteringParams): void {
    const {
      n_clusters,
      distance_threshold,
      linkage = 'ward',
      metric = 'euclidean',
    } = params;

    const has_n_clusters = n_clusters != null;
    const has_threshold = distance_threshold != null;

    if (has_n_clusters === has_threshold) {
      throw new Error(
        'Provide exactly one of n_clusters or distance_threshold.',
      );
    }

    if (has_n_clusters && (!Number.isInteger(n_clusters) || n_clusters < 1)) {
      throw new Error('n_clusters must be a positive integer (>= 1).');
    }

    if (has_threshold && !(distance_threshold > 0)) {
      throw new Error('distance_threshold must be a positive number.');
    }

    if (!AgglomerativeClustering.VALID_LINKAGES.includes(linkage)) {
      throw new Error(
        `Invalid linkage '${linkage}'. Must be one of ${AgglomerativeClustering.VALID_LINKAGES.join(', ')}.`,
      );
    }

    if (!AgglomerativeClustering.VALID_METRICS.includes(metric)) {
      throw new Error(
        `Invalid metric '${metric}'. Must be one of ${AgglomerativeClustering.VALID_METRICS.join(', ')}.`,
      );
    }

    if (linkage === 'ward' && metric !== 'euclidean') {
      throw new Error("Ward linkage requires metric to be 'euclidean'.");
    }
  }

  /**
   * Computes the full `n×n` pairwise distance matrix in float64, laid out
   * row-major (`i*n+j`) to match the NN-chain engine.
   *
   * **Symmetry**: for every `(i, j)` pair the same scalar `dist` is assigned
   * to both `D[i*n+j]` and `D[j*n+i]`, so `D[i,j] ≡ D[j,i]` exactly —
   * not approximately.  `(D+Dᵀ)/2` symmetrisation is unnecessary here
   * because asymmetry cannot arise from this mirror pattern.
   *
   * **Zero diagonal**: `Float64Array` zero-initialises; the inner loop
   * iterates `j > i` and never writes `D[i*n+i]`.
   *
   * **float64 only**: `pairwise_distance_matrix` in `distance/pairwise_distance.ts`
   * operates on float32 TensorFlow tensors and applies Gram-matrix shortcuts
   * that require clamping and explicit symmetrisation.  This method uses
   * direct scalar arithmetic at float64 precision, which the NN-chain
   * tie-resolution requires to reproduce sklearn's merge order exactly.
   * Delegating to the tensor path would silently degrade precision.
   *
   * Distances use the same definitions as scikit-learn's `pairwise_distances`,
   * so results are bit-identical to sklearn for the same coordinates.
   */
  private static compute_distance_matrix(
    data: number[][],
    metric: 'euclidean' | 'manhattan' | 'cosine',
  ): Float64Array {
    const n = data.length;
    const dim = n > 0 ? data[0].length : 0;
    const D = new Float64Array(n * n);

    if (metric === 'euclidean') {
      for (let i = 0; i < n; i++) {
        const ri = data[i];
        for (let j = i + 1; j < n; j++) {
          const rj = data[j];
          let sum = 0;
          for (let k = 0; k < dim; k++) {
            const diff = ri[k] - rj[k];
            sum += diff * diff;
          }
          const dist = Math.sqrt(sum);
          D[i * n + j] = dist;
          D[j * n + i] = dist;
        }
      }
    } else if (metric === 'manhattan') {
      for (let i = 0; i < n; i++) {
        const ri = data[i];
        for (let j = i + 1; j < n; j++) {
          const rj = data[j];
          let sum = 0;
          for (let k = 0; k < dim; k++) {
            sum += Math.abs(ri[k] - rj[k]);
          }
          D[i * n + j] = sum;
          D[j * n + i] = sum;
        }
      }
    } else {
      // cosine: 1 − (xᵢ·xⱼ) / (‖xᵢ‖·‖xⱼ‖); a zero-norm vector yields distance 0.
      const norms = new Float64Array(n);
      for (let i = 0; i < n; i++) {
        const ri = data[i];
        let sq = 0;
        for (let k = 0; k < dim; k++) sq += ri[k] * ri[k];
        norms[i] = Math.sqrt(sq);
      }
      for (let i = 0; i < n; i++) {
        const ri = data[i];
        for (let j = i + 1; j < n; j++) {
          const rj = data[j];
          let dot = 0;
          for (let k = 0; k < dim; k++) dot += ri[k] * rj[k];
          const denom = norms[i] * norms[j];
          const dist = denom === 0 ? 0 : 1 - dot / denom;
          D[i * n + j] = dist;
          D[j * n + i] = dist;
        }
      }
    }

    return D;
  }

  /**
   * Converts raw active-slot merge records into sklearn/scipy-style children
   * node ids (`0..n-1` leaves, `n..` internal nodes).
   */
  private static build_children(
    merges: MergeRecord[],
    n_samples: number,
  ): number[][] {
    const parent = new Int32Array(2 * n_samples - 1);
    for (let i = 0; i < parent.length; i++) {
      parent[i] = i;
    }

    function find(x: number): number {
      while (parent[x] !== x) {
        parent[x] = parent[parent[x]];
        x = parent[x];
      }
      return x;
    }

    const children: number[][] = [];
    let next_node = n_samples;

    for (const merge of merges) {
      const left = find(merge.cluster_a);
      const right = find(merge.cluster_b);
      children.push([Math.min(left, right), Math.max(left, right)]);
      parent[left] = next_node;
      parent[right] = next_node;
      next_node++;
    }

    return children;
  }

  /**
   * Validates that a precomputed distance matrix is square, symmetric, and has
   * a zero diagonal.
   */
  private static validate_precomputed(D: number[][]): void {
    const n = D.length;
    if (n === 0) {
      throw new Error(
        'Precomputed distance matrix must contain at least one row.',
      );
    }

    for (let i = 0; i < n; i++) {
      if (!Array.isArray(D[i]) || D[i].length !== n) {
        throw new Error(
          `Precomputed distance matrix must be square (${n}x${n}).`,
        );
      }
    }

    const tol = 1e-8;
    for (let i = 0; i < n; i++) {
      if (Math.abs(D[i][i]) > tol) {
        throw new Error(
          'Precomputed distance matrix must have a zero diagonal.',
        );
      }
      for (let j = i + 1; j < n; j++) {
        if (Math.abs(D[i][j] - D[j][i]) > tol) {
          throw new Error('Precomputed distance matrix must be symmetric.');
        }
      }
    }
  }
}
