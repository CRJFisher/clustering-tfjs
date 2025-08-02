import type {
  DataMatrix,
  LabelVector,
  AgglomerativeClusteringParams,
  BaseClustering,
} from './types';
import tf from '../tf-adapter';
import { pairwiseDistanceMatrix } from '../utils/pairwise_distance';
import { update_distance_matrix } from './linkage';

/**
 * Agglomerative (hierarchical) clustering estimator skeleton.
 *
 * Only the constructor, parameter validation and public property definitions
 * are implemented as part of this initial task. The actual clustering logic
 * will be added in subsequent tasks.
 */
export class AgglomerativeClustering
  implements BaseClustering<AgglomerativeClusteringParams>
{
  /**
   * Hyper-parameters describing the behaviour of this instance.
   */
  public readonly params: AgglomerativeClusteringParams;

  /**
   * Cluster labels produced by `fit` / `fitPredict`.
   *
   * Populated after calling `fit`.
   */
  public labels_: LabelVector | null = null;

  /**
   * Children of each non-leaf node in the hierarchical clustering tree.
   * Shape: `(nSamples-1, 2)` where each row gives the indices of the merged
   * clusters. Lazily populated by future implementation.
   */
  public children_: number[][] | null = null;

  /**
   * Number of leaves in the hierarchical clustering tree (equals `nSamples`).
   */
  public nLeaves_: number | null = null;

  /**
   * Allowed linkage strategies.
   */
  private static readonly VALID_LINKAGES = [
    'ward',
    'complete',
    'average',
    'single',
  ] as const;

  /**
   * Allowed distance metrics.
   */
  private static readonly VALID_METRICS = [
    'euclidean',
    'manhattan',
    'cosine',
  ] as const;

  constructor(params: AgglomerativeClusteringParams) {
    // Perform a shallow copy to freeze user input and avoid side effects.
    this.params = { ...params };

    AgglomerativeClustering.validateParams(this.params);
  }

  /**
   * Fits the estimator to the provided data matrix.
   *
   * Note: The actual algorithm is not implemented yet. The stub only exists so
   * the public interface is complete and unit tests can assert that the method
   * is callable.
   */
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  async fit(_X: DataMatrix): Promise<void> {
    // Convert input to a tf.Tensor2D for distance computation if necessary.

    // Early exit for edge-cases ------------------------------------------------
    if (Array.isArray(_X) && _X.length === 0) {
      throw new Error('Input X must contain at least one sample.');
    }

    const points: tf.Tensor2D = Array.isArray(_X)
      ? tf.tensor2d(_X as number[][])
      : (_X as tf.Tensor2D);

    const nSamples = points.shape[0];

    // Handle trivial case of single sample separately
    if (nSamples === 1) {
      this.labels_ = [0];
      this.children_ = [];
      this.nLeaves_ = 1;
      points.dispose?.();
      return;
    }

    const { metric = 'euclidean', linkage = 'ward', nClusters } = this.params;

    // -----------------------------------------------------------------------
    // Compute initial pairwise distance matrix (plain number[][] for fast JS
    // level manipulation). We leverage the existing helper in utils.
    // -----------------------------------------------------------------------
    const distanceTensor = pairwiseDistanceMatrix(points, metric);
    const D: number[][] = (await distanceTensor.array()) as number[][];
    distanceTensor.dispose();

    /*  ------------------------------------------------------------------
     *  Hierarchical agglomeration loop
     *  ------------------------------------------------------------------ */

    // Cluster bookkeeping arrays. Index i corresponds to row/col i in D.
    const clusterIds: number[] = Array.from({ length: nSamples }, (_, i) => i);
    const clusterSizes: number[] = Array(nSamples).fill(1);
    let nextClusterId = nSamples; // new clusters get incremental ids

    const children: number[][] = [];

    // Track current cluster label for each sample (global cluster ids)
    const sampleLabels: number[] = Array.from(
      { length: nSamples },
      (_, i) => i,
    );

    // Merge until the desired number of clusters is reached.
    while (clusterIds.length > nClusters) {
      // -------------------------------------------------------------------
      // Find closest pair (i,j)
      // -------------------------------------------------------------------
      let minDist = Number.POSITIVE_INFINITY;
      let minI = 0;
      let minJ = 1;

      for (let i = 0; i < D.length; i++) {
        for (let j = i + 1; j < D.length; j++) {
          const d = D[i][j];
          if (d < minDist) {
            minDist = d;
            minI = i;
            minJ = j;
          }
        }
      }

      // Store merge in children_ (using global cluster ids)
      const idI = clusterIds[minI];
      const idJ = clusterIds[minJ];
      children.push([idI, idJ]);

      // Update distance matrix & auxiliary arrays
      update_distance_matrix(D, clusterSizes, minI, minJ, linkage);

      // Assign a new cluster id to the merged entity (row minI after update)
      const newId = nextClusterId++;
      clusterIds[minI] = newId;
      clusterIds.splice(minJ, 1);

      // Propagate new labels to samples that belonged to idI or idJ
      for (let s = 0; s < nSamples; s++) {
        const lbl = sampleLabels[s];
        if (lbl === idI || lbl === idJ) {
          sampleLabels[s] = newId;
        }
      }

      // Loop continues with contracted D.
    }

    // ---------------------------------------------------------------------
    // Derive flat cluster labels by cutting dendrogram at desired number of
    // clusters. The simplest approach is to recreate cluster membership from
    // bottom-up using the recorded merges.
    // ---------------------------------------------------------------------

    const labels = sampleLabels;
    // Relabel to contiguous range 0 .. nClusters-1
    const uniqueOld = Array.from(new Set(labels));
    const mapping = new Map<number, number>();
    uniqueOld.forEach((oldLabel, newLabel) => mapping.set(oldLabel, newLabel));
    this.labels_ = labels.map((old) => mapping.get(old)!) as number[];

    this.children_ = children;
    this.nLeaves_ = nSamples;

    // Dispose created tensor if we have created one from array input.
    if (Array.isArray(_X)) {
      points.dispose();
    }
  }

  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  async fitPredict(_X: DataMatrix): Promise<LabelVector> {
    await this.fit(_X);
    if (this.labels_ == null) {
      throw new Error('AgglomerativeClustering failed to compute labels.');
    }
    return this.labels_;
  }

  /* --------------------------------------------------------------------- */
  /*                         Parameter Validation                          */
  /* --------------------------------------------------------------------- */

  private static validateParams(params: AgglomerativeClusteringParams): void {
    const { nClusters, linkage = 'ward', metric = 'euclidean' } = params;

    // nClusters must be a positive integer
    if (!Number.isInteger(nClusters) || nClusters < 1) {
      throw new Error('nClusters must be a positive integer (>= 1).');
    }

    // linkage value
    if (!AgglomerativeClustering.VALID_LINKAGES.includes(linkage)) {
      throw new Error(
        `Invalid linkage '${linkage}'. Must be one of ${AgglomerativeClustering.VALID_LINKAGES.join(', ')}.`,
      );
    }

    // metric value
    if (!AgglomerativeClustering.VALID_METRICS.includes(metric)) {
      throw new Error(
        `Invalid metric '${metric}'. Must be one of ${AgglomerativeClustering.VALID_METRICS.join(', ')}.`,
      );
    }

    // Additional consistency check: Ward linkage requires Euclidean distance.
    if (linkage === 'ward' && metric !== 'euclidean') {
      throw new Error("Ward linkage requires metric to be 'euclidean'.");
    }
  }
}
