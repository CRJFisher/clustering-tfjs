import type {
  DataMatrix,
  AgglomerativeClusteringParams,
  BaseClustering,
} from './types';
import * as tf from '../tf-adapter';
import { isTensor } from '../utils/tensor-utils';
import { pairwiseDistanceMatrix } from '../utils/pairwise_distance';
import { storedNNCluster } from './linkage';

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
   * Cluster labels produced by `fit` / `fitPredict`.
   *
   * Populated after calling `fit`.
   */
  public labels_: number[] | null = null;

  /**
   * Children of each non-leaf node in the hierarchical clustering tree.
   * Shape: `(nSamples-1, 2)` where each row gives the indices of the merged
   * clusters. Lazily populated by future implementation.
   */
  public children_: number[][] | null = null;
  public nLeaves_: number | null = null;

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
    AgglomerativeClustering.validateParams(this.params);
  }

  /**
   * Fits the agglomerative clustering model to the input data.
   *
   * @param _X - Input data matrix of shape [nSamples, nFeatures].
   * @returns A promise that resolves when fitting is complete.
   * @throws {Error} If input is empty or nClusters exceeds nSamples.
   *
   * @example
   * ```typescript
   * const agg = new AgglomerativeClustering({ nClusters: 3 });
   * await agg.fit([[1, 2], [3, 4], [5, 6]]);
   * console.log(agg.labels_);
   * ```
   */
  async fit(_X: DataMatrix): Promise<void> {
    const ownedTensor = !isTensor(_X);
    const points: tf.Tensor2D = ownedTensor
      ? tf.tensor2d(_X as number[][])
      : (_X as tf.Tensor2D);

    const nSamples = points.shape[0];

    if (nSamples === 0) {
      if (ownedTensor) {
        points.dispose();
      }
      throw new Error('Input X must contain at least one sample.');
    }

    if (this.params.nClusters > nSamples) {
      if (ownedTensor) {
        points.dispose();
      }
      throw new Error('nClusters cannot exceed number of samples.');
    }

    // Handle trivial case of single sample separately
    if (nSamples === 1) {
      this.labels_ = [0];
      this.children_ = [];
      this.nLeaves_ = 1;
      if (ownedTensor) {
        points.dispose();
      }
      return;
    }

    const { metric = 'euclidean', linkage = 'ward', nClusters } = this.params;

    // Compute initial pairwise distance matrix
    const distanceTensor = pairwiseDistanceMatrix(points, metric);
    const D2d = (await distanceTensor.array()) as number[][];
    distanceTensor.dispose();

    // Convert to flat Float64Array for cache-friendly access and in-place updates
    const D = new Float64Array(nSamples * nSamples);
    for (let i = 0; i < nSamples; i++) {
      for (let j = 0; j < nSamples; j++) {
        D[i * nSamples + j] = D2d[i][j];
      }
    }

    // Run stored-nearest-neighbor clustering — O(n²) amortized
    const merges = storedNNCluster(D, nSamples, nClusters, linkage);

    // Build children_ array using global cluster IDs (sklearn convention:
    // original samples are 0..n-1, each merge creates n, n+1, n+2, ...)
    const globalId = new Int32Array(nSamples);
    for (let i = 0; i < nSamples; i++) globalId[i] = i;
    let nextGlobalId = nSamples;

    const children: number[][] = [];
    for (const m of merges) {
      children.push([globalId[m.clusterA], globalId[m.clusterB]]);
      globalId[m.clusterA] = nextGlobalId++;
    }

    // Derive flat cluster labels using Union-Find over the merge history
    const parent = new Int32Array(nSamples);
    for (let i = 0; i < nSamples; i++) parent[i] = i;

    function find(x: number): number {
      while (parent[x] !== x) {
        parent[x] = parent[parent[x]]; // path compression
        x = parent[x];
      }
      return x;
    }

    // Replay all merges — the survivor absorbs the removed cluster
    for (const m of merges) {
      const ra = find(m.clusterA);
      const rb = find(m.clusterB);
      parent[rb] = ra;
    }

    // Assign contiguous labels 0..nClusters-1
    const labels = new Array<number>(nSamples);
    const rootToLabel = new Map<number, number>();
    let nextLabel = 0;
    for (let i = 0; i < nSamples; i++) {
      const root = find(i);
      if (!rootToLabel.has(root)) {
        rootToLabel.set(root, nextLabel++);
      }
      labels[i] = rootToLabel.get(root)!;
    }

    this.labels_ = labels;
    this.children_ = children;
    this.nLeaves_ = nSamples;

    // Dispose tensor if we created it from array input
    if (ownedTensor) {
      points.dispose();
    }
  }

  /**
   * Fits the model and returns cluster labels.
   *
   * @param _X - Input data matrix of shape [nSamples, nFeatures].
   * @returns Array of cluster labels for each sample.
   * @throws {Error} If input is empty or nClusters exceeds nSamples.
   */
  async fitPredict(_X: DataMatrix): Promise<number[]> {
    await this.fit(_X);
    if (this.labels_ == null) {
      throw new Error('AgglomerativeClustering failed to compute labels.');
    }
    return this.labels_;
  }

  private static validateParams(params: AgglomerativeClusteringParams): void {
    const { nClusters, linkage = 'ward', metric = 'euclidean' } = params;

    if (!Number.isInteger(nClusters) || nClusters < 1) {
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
