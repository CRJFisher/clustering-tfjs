import type {
  DataMatrix,
  AgglomerativeClusteringParams,
  BaseClustering,
} from './types';
import * as tf from '../backend/adapter';
import { is_tensor } from '../tensor/tensor_guards';
import { pairwise_distance_matrix } from '../distance/pairwise_distance';
import { stored_nn_cluster } from './linkage';

/**
 * Agglomerative (hierarchical) clustering using a stored-nearest-neighbor
 * merge strategy with Lance–Williams distance updates.
 *
 * Achieves O(n²) amortized complexity instead of the naive O(n³) approach by
 * caching per-cluster nearest neighbors and updating them incrementally.
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
   * Children of each non-leaf node in the hierarchical clustering tree.
   * Shape: `(n_samples-1, 2)` where each row gives the indices of the merged
   * clusters. Lazily populated by future implementation.
   */
  public children_: number[][] | null = null;
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
    const owned_tensor = !is_tensor(_X);
    const points: tf.Tensor2D = owned_tensor
      ? tf.tensor2d(_X as number[][])
      : (_X as tf.Tensor2D);

    const n_samples = points.shape[0];

    if (n_samples === 0) {
      if (owned_tensor) {
        points.dispose();
      }
      throw new Error('Input X must contain at least one sample.');
    }

    if (this.params.n_clusters > n_samples) {
      if (owned_tensor) {
        points.dispose();
      }
      throw new Error('nClusters cannot exceed number of samples.');
    }

    // Handle trivial case of single sample separately
    if (n_samples === 1) {
      this.labels_ = [0];
      this.children_ = [];
      this.n_leaves_ = 1;
      if (owned_tensor) {
        points.dispose();
      }
      return;
    }

    const { metric = 'euclidean', linkage = 'ward', n_clusters } = this.params;

    // Compute initial pairwise distance matrix
    const distance_tensor = pairwise_distance_matrix(points, metric);
    const D2d = (await distance_tensor.array()) as number[][];
    distance_tensor.dispose();

    // Convert to flat Float64Array for cache-friendly access and in-place updates
    const D = new Float64Array(n_samples * n_samples);
    for (let i = 0; i < n_samples; i++) {
      for (let j = 0; j < n_samples; j++) {
        D[i * n_samples + j] = D2d[i][j];
      }
    }

    // Run stored-nearest-neighbor clustering — O(n²) amortized
    const merges = stored_nn_cluster(D, n_samples, n_clusters, linkage);

    // Build children_ array using global cluster IDs (sklearn convention:
    // original samples are 0..n-1, each merge creates n, n+1, n+2, ...)
    const global_id = new Int32Array(n_samples);
    for (let i = 0; i < n_samples; i++) global_id[i] = i;
    let next_global_id = n_samples;

    const children: number[][] = [];
    for (const m of merges) {
      children.push([global_id[m.cluster_a], global_id[m.cluster_b]]);
      global_id[m.cluster_a] = next_global_id++;
    }

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
    this.n_leaves_ = n_samples;

    // Dispose tensor if we created it from array input
    if (owned_tensor) {
      points.dispose();
    }
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
    const { n_clusters, linkage = 'ward', metric = 'euclidean' } = params;

    if (!Number.isInteger(n_clusters) || n_clusters < 1) {
      throw new Error('nClusters must be a positive integer (>= 1).');
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
}
