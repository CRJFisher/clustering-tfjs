import type {
  DataMatrix,
  AgglomerativeClusteringParams,
  BaseClustering,
} from './types';
import * as tf from '../backend/adapter';
import { is_tensor } from '../tensor/tensor_guards';
import { pairwise_distance_matrix } from '../distance/pairwise_distance';
import { MergeRecord, nn_chain_cluster } from './linkage';

/**
 * Agglomerative (hierarchical) clustering using nearest-neighbor chain merges
 * with Lance–Williams distance updates.
 *
 * Builds the full reducible-linkage tree in O(n²) time for single, complete,
 * average, and Ward linkage, then cuts that tree by `n_clusters` or
 * `distance_threshold`.
 */
export class AgglomerativeClustering
  implements BaseClustering<AgglomerativeClusteringParams>
{
  public readonly params: AgglomerativeClusteringParams;

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
      const owned_tensor = !is_tensor(_X);
      const points: tf.Tensor2D = owned_tensor
        ? tf.tensor2d(_X as number[][])
        : (_X as tf.Tensor2D);

      n_samples = points.shape[0];

      if (n_samples === 0) {
        if (owned_tensor) {
          points.dispose();
        }
        throw new Error('Input X must contain at least one sample.');
      }

      // Compute initial pairwise distance matrix. Read it as a flat row-major
      // typed array (matching the i*n+j layout used downstream) and copy into a
      // Float64Array for cache-friendly access and in-place updates — avoiding
      // the heavy intermediate nested number[][].
      const distance_tensor = pairwise_distance_matrix(points, metric);
      const flat = await distance_tensor.data();
      distance_tensor.dispose();

      D = new Float64Array(n_samples * n_samples);
      D.set(flat);

      if (owned_tensor) {
        points.dispose();
      }
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
