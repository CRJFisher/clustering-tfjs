import type {
  DataMatrix,
  LabelVector,
  AgglomerativeClusteringParams,
  BaseClustering,
} from './types';
import * as tf from '../tf-adapter';
import { pairwiseDistanceMatrix } from '../utils/pairwise_distance';
import { heapCluster } from './linkage';

/**
 * Agglomerative (hierarchical) clustering using a priority-queue-based
 * merge strategy with Lance–Williams distance updates.
 *
 * Achieves O(n² log n) complexity instead of the naive O(n³) approach by
 * maintaining a min-heap of candidate merge pairs with lazy deletion.
 */
export class AgglomerativeClustering
  implements BaseClustering<AgglomerativeClusteringParams>
{
  public readonly params: AgglomerativeClusteringParams;
  public labels_: LabelVector | null = null;
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

  constructor(params: AgglomerativeClusteringParams) {
    this.params = { ...params };
    AgglomerativeClustering.validateParams(this.params);
  }

  async fit(_X: DataMatrix): Promise<void> {
    if (Array.isArray(_X) && _X.length === 0) {
      throw new Error('Input X must contain at least one sample.');
    }

    const points: tf.Tensor2D = Array.isArray(_X)
      ? tf.tensor2d(_X)
      : (_X as tf.Tensor2D);

    const nSamples = points.shape[0];

    // Handle trivial case of single sample separately
    if (nSamples === 1) {
      this.labels_ = [0];
      this.children_ = [];
      this.nLeaves_ = 1;
      if (Array.isArray(_X)) {
        points.dispose();
      }
      return;
    }

    const { metric = 'euclidean', linkage = 'ward', nClusters } = this.params;

    // Compute initial pairwise distance matrix
    const distanceTensor = pairwiseDistanceMatrix(points, metric);
    const D2d: number[][] = (await distanceTensor.array()) as number[][];
    distanceTensor.dispose();

    // Convert to flat Float64Array for cache-friendly access and in-place updates
    const D = new Float64Array(nSamples * nSamples);
    for (let i = 0; i < nSamples; i++) {
      for (let j = 0; j < nSamples; j++) {
        D[i * nSamples + j] = D2d[i][j];
      }
    }

    // Run priority-queue-based clustering — O(n² log n)
    const merges = heapCluster(D, nSamples, nClusters, linkage);

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
    if (Array.isArray(_X)) {
      points.dispose();
    }
  }

  async fitPredict(_X: DataMatrix): Promise<LabelVector> {
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
