import * as tf from '../tf-adapter';
import { KMeans } from '../clustering/kmeans';
import { SpectralClustering } from '../clustering/spectral';
import { AgglomerativeClustering } from '../clustering/agglomerative';
import { SOM } from '../clustering/som';
import { silhouetteScore } from '../validation/silhouette';
import { daviesBouldinEfficient } from '../validation/davies_bouldin';
import { calinskiHarabaszEfficient } from '../validation/calinski_harabasz';
import { isTensor } from './tensor-utils';
import { computeWss } from './computeWss';
import { findKnee } from './kneedle';
import type { DataMatrix } from '../clustering/types';

/**
 * Result for a single k value evaluation
 */
export interface ClusterEvaluation {
  /** Number of clusters */
  k: number;
  /** Silhouette score (range: [-1, 1], higher is better) */
  silhouette: number;
  /** Davies-Bouldin index (range: [0, ∞), lower is better) */
  daviesBouldin: number;
  /** Calinski-Harabasz index (range: [0, ∞), higher is better) */
  calinskiHarabasz: number;
  /**
   * Combined score used for selection.
   * When using the default 'combined' method, this is the mean of normalized
   * metrics in [0, 1]. When using 'silhouette', this is the raw silhouette.
   * When using 'elbow', the knee point gets score 1.0.
   */
  combinedScore: number;
  /** Cluster labels for this k */
  labels: number[];
  /** Within-cluster sum of squares (inertia). Present when method is 'elbow'. */
  wss?: number;
}

/**
 * Method for selecting the optimal number of clusters.
 * - 'combined': Normalized combination of silhouette, Calinski-Harabasz, and Davies-Bouldin
 * - 'elbow': WSS/inertia curve knee detection
 * - 'silhouette': Highest silhouette score
 */
export type OptimalClustersMethod = 'combined' | 'elbow' | 'silhouette';

/**
 * Options for finding optimal clusters
 */
export interface FindOptimalClustersOptions {
  /** Minimum number of clusters to test (default: 2) */
  minClusters?: number;
  /** Maximum number of clusters to test (default: 10) */
  maxClusters?: number;
  /** Algorithm to use (default: 'kmeans') */
  algorithm?: 'kmeans' | 'spectral' | 'agglomerative' | 'som';
  /** Algorithm-specific parameters */
  algorithmParams?: Record<string, unknown>;
  /** Metrics to use for evaluation (default: all). Only used with 'combined' method. */
  metrics?: Array<'silhouette' | 'daviesBouldin' | 'calinskiHarabasz'>;
  /**
   * Custom scoring function. Receives raw (un-normalized) metric values.
   * Overrides the `method` option when provided.
   */
  scoringFunction?: (evaluation: ClusterEvaluation) => number;
  /** Method for selecting optimal k (default: 'combined') */
  method?: OptimalClustersMethod;
}

/**
 * Normalizes metrics across all evaluations and computes combined scores.
 * Each metric is normalized to [0, 1] and averaged for equal contribution.
 */
function normalizeAndScoreEvaluations(
  evaluations: ClusterEvaluation[],
  metrics: Array<'silhouette' | 'daviesBouldin' | 'calinskiHarabasz'>,
): void {
  if (evaluations.length === 0) return;

  // Pre-compute min-max for Calinski-Harabasz (higher is better)
  let chMin = Infinity;
  let chMax = -Infinity;
  if (metrics.includes('calinskiHarabasz')) {
    for (const e of evaluations) {
      if (Number.isFinite(e.calinskiHarabasz)) {
        chMin = Math.min(chMin, e.calinskiHarabasz);
        chMax = Math.max(chMax, e.calinskiHarabasz);
      }
    }
  }
  const chRange = chMax - chMin;

  // Pre-compute min-max for Davies-Bouldin (lower is better)
  let dbMin = Infinity;
  let dbMax = -Infinity;
  if (metrics.includes('daviesBouldin')) {
    for (const e of evaluations) {
      if (Number.isFinite(e.daviesBouldin)) {
        dbMin = Math.min(dbMin, e.daviesBouldin);
        dbMax = Math.max(dbMax, e.daviesBouldin);
      }
    }
  }
  const dbRange = dbMax - dbMin;

  // Score each evaluation with normalized metrics
  for (const evaluation of evaluations) {
    let score = 0;
    let numMetrics = 0;

    if (metrics.includes('silhouette')) {
      // Fixed range normalization: [-1, 1] -> [0, 1]
      const ns = (evaluation.silhouette + 1) / 2;
      score += Number.isFinite(ns) ? ns : 0;
      numMetrics++;
    }

    if (metrics.includes('calinskiHarabasz')) {
      let nch: number;
      if (!Number.isFinite(evaluation.calinskiHarabasz)) {
        nch = 0;
      } else if (chRange === 0 || !Number.isFinite(chRange)) {
        nch = 0.5;
      } else {
        nch = (evaluation.calinskiHarabasz - chMin) / chRange;
      }
      score += nch;
      numMetrics++;
    }

    if (metrics.includes('daviesBouldin')) {
      let ndb: number;
      if (!Number.isFinite(evaluation.daviesBouldin)) {
        ndb = 0;
      } else if (dbRange === 0 || !Number.isFinite(dbRange)) {
        ndb = 0.5;
      } else {
        ndb = 1 - (evaluation.daviesBouldin - dbMin) / dbRange;
      }
      score += ndb;
      numMetrics++;
    }

    evaluation.combinedScore = numMetrics > 0 ? score / numMetrics : 0;
  }
}

/**
 * Automatically finds the optimal number of clusters for a dataset by evaluating
 * multiple k values using validation metrics.
 *
 * @param X - Input data matrix (samples × features)
 * @param options - Configuration options
 * @returns Object containing optimal k and detailed results for all tested k values
 *
 * @example
 * ```typescript
 * import { findOptimalClusters } from 'clustering-tfjs';
 *
 * const data = [[1, 2], [1.5, 1.8], [5, 8], [8, 8], [1, 0.6], [9, 11]];
 * const result = await findOptimalClusters(data, { maxClusters: 5 });
 *
 * console.log(`Optimal number of clusters: ${result.optimal.k}`);
 * console.log(`Best silhouette score: ${result.optimal.silhouette}`);
 * ```
 */
export async function findOptimalClusters(
  X: DataMatrix,
  options: FindOptimalClustersOptions = {},
): Promise<{
  /** The optimal cluster evaluation */
  optimal: ClusterEvaluation;
  /** All evaluations sorted by combined score (descending) */
  evaluations: ClusterEvaluation[];
}> {
  const {
    minClusters = 2,
    maxClusters = 10,
    algorithm = 'kmeans',
    algorithmParams = {},
    metrics = ['silhouette', 'daviesBouldin', 'calinskiHarabasz'],
    scoringFunction,
    method = 'combined',
  } = options;

  // Validate inputs
  if (minClusters < 2) {
    throw new Error('minClusters must be at least 2');
  }
  if (maxClusters < minClusters) {
    throw new Error('maxClusters must be greater than or equal to minClusters');
  }

  // Convert data to tensor if needed
  const isInputTensor = isTensor(X);
  const dataTensor = isInputTensor ? X : tf.tensor2d(X as number[][]);
  const nSamples = dataTensor.shape[0];

  // Adjust maxClusters if it exceeds number of samples
  const effectiveMaxClusters = Math.min(maxClusters, nSamples - 1);

  if (effectiveMaxClusters < minClusters) {
    // Clean up tensor if we created it
    if (!isInputTensor) {
      dataTensor.dispose();
    }
    throw new Error(
      `Not enough samples (${nSamples}) for minimum clusters (${minClusters})`,
    );
  }

  // Determine which metrics to compute based on method
  const computeSilhouette =
    method === 'silhouette' ||
    method === 'combined' && metrics.includes('silhouette') ||
    !!scoringFunction && metrics.includes('silhouette');
  const computeDB =
    method === 'combined' && metrics.includes('daviesBouldin') ||
    !!scoringFunction && metrics.includes('daviesBouldin');
  const computeCH =
    method === 'combined' && metrics.includes('calinskiHarabasz') ||
    !!scoringFunction && metrics.includes('calinskiHarabasz');
  const computeWSS = method === 'elbow';

  const evaluations: ClusterEvaluation[] = [];

  // Test each k value
  for (let k = minClusters; k <= effectiveMaxClusters; k++) {
    // Create clustering instance
    let clusterer;
    let kmeansInstance: KMeans | null = null;

    switch (algorithm) {
      case 'kmeans':
        kmeansInstance = new KMeans({ nClusters: k, ...algorithmParams });
        clusterer = kmeansInstance;
        break;
      case 'spectral':
        clusterer = new SpectralClustering({
          nClusters: k,
          ...algorithmParams,
        });
        break;
      case 'agglomerative':
        clusterer = new AgglomerativeClustering({
          nClusters: k,
          ...algorithmParams,
        });
        break;
      case 'som': {
        // For SOM, we need to determine grid dimensions
        // Use square grid as default, can be overridden via algorithmParams
        const gridSize = Math.ceil(Math.sqrt(k));
        const params = algorithmParams as Record<string, unknown>;
        clusterer = new SOM({
          nClusters: k,
          gridWidth: (params.gridWidth as number) || gridSize,
          gridHeight: (params.gridHeight as number) || Math.ceil(k / gridSize),
          ...algorithmParams,
        });
        break;
      }
      default:
        throw new Error(`Unknown algorithm: ${algorithm}`);
    }

    // Fit and predict
    const labelsTensor = await clusterer.fitPredict(dataTensor);

    // Convert labels to array if it's a tensor
    const labels =
      isTensor(labelsTensor)
        ? ((await labelsTensor.data()) as unknown as number[])
        : labelsTensor;

    // Calculate metrics
    let silhouette = 0;
    let daviesBouldin = Infinity;
    let calinskiHarabasz = 0;
    let wss: number | undefined;

    if (computeSilhouette) {
      silhouette = silhouetteScore(dataTensor, labels);
    }
    if (computeDB) {
      daviesBouldin = daviesBouldinEfficient(dataTensor, labels);
    }
    if (computeCH) {
      calinskiHarabasz = calinskiHarabaszEfficient(dataTensor, labels);
    }
    if (computeWSS) {
      // Optimize: read inertia directly from KMeans instead of recomputing
      if (kmeansInstance && kmeansInstance.inertia_ !== null) {
        wss = kmeansInstance.inertia_;
      } else {
        wss = computeWss(dataTensor, labels);
      }
    }

    // Build evaluation
    const evaluation: ClusterEvaluation = {
      k,
      silhouette,
      daviesBouldin,
      calinskiHarabasz,
      combinedScore: 0,
      labels: Array.from(labels),
    };
    if (wss !== undefined) {
      evaluation.wss = wss;
    }

    // Dispose labels tensor if needed
    if (isTensor(labelsTensor)) {
      labelsTensor.dispose();
    }

    // Apply custom scoring function immediately (gets raw values)
    if (scoringFunction) {
      evaluation.combinedScore = scoringFunction(evaluation);
    }

    evaluations.push(evaluation);
  }

  // Post-loop scoring based on method (only when no custom scorer)
  if (!scoringFunction) {
    if (method === 'combined') {
      normalizeAndScoreEvaluations(evaluations, metrics);
    } else if (method === 'silhouette') {
      for (const evaluation of evaluations) {
        evaluation.combinedScore = evaluation.silhouette;
      }
    } else if (method === 'elbow') {
      const kValues = evaluations.map((e) => e.k);
      const wssValues = evaluations.map((e) => e.wss!);
      const result = findKnee(kValues, wssValues, { direction: 'concave' });

      if (result.kneeX !== null) {
        // Use difference values as scores (higher = better elbow candidate)
        for (let i = 0; i < evaluations.length; i++) {
          evaluations[i].combinedScore = result.differences[i];
        }
      } else {
        // Fallback: prefer smallest k (parsimony)
        for (const evaluation of evaluations) {
          evaluation.combinedScore = -evaluation.k;
        }
      }
    }
  }

  // Sort by combined score (descending)
  evaluations.sort((a, b) => b.combinedScore - a.combinedScore);

  // Store result before cleanup
  const finalResult = {
    optimal: evaluations[0],
    evaluations,
  };

  // Clean up tensor if we created it
  if (!isInputTensor) {
    dataTensor.dispose();
  }

  return finalResult;
}
