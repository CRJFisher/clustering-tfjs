import * as tf from "@tensorflow/tfjs-node";
import { KMeans } from "../clustering/kmeans";
import { SpectralClustering } from "../clustering/spectral";
import { AgglomerativeClustering } from "../clustering/agglomerative";
import { silhouetteScore } from "../validation/silhouette";
import { daviesBouldinEfficient } from "../validation/davies_bouldin";
import { calinskiHarabaszEfficient } from "../validation/calinski_harabasz";
import type { DataMatrix } from "../clustering/types";

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
  /** Combined score used for selection */
  combinedScore: number;
  /** Cluster labels for this k */
  labels: number[];
}

/**
 * Options for finding optimal clusters
 */
export interface FindOptimalClustersOptions {
  /** Minimum number of clusters to test (default: 2) */
  minClusters?: number;
  /** Maximum number of clusters to test (default: 10) */
  maxClusters?: number;
  /** Algorithm to use (default: 'kmeans') */
  algorithm?: 'kmeans' | 'spectral' | 'agglomerative';
  /** Algorithm-specific parameters */
  algorithmParams?: Record<string, unknown>;
  /** Metrics to use for evaluation (default: all) */
  metrics?: Array<'silhouette' | 'daviesBouldin' | 'calinskiHarabasz'>;
  /** Custom scoring function (default: silhouette + calinski - davies) */
  scoringFunction?: (evaluation: ClusterEvaluation) => number;
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
 * import { findOptimalClusters } from 'clustering-js';
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
  options: FindOptimalClustersOptions = {}
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
    scoringFunction
  } = options;

  // Validate inputs
  if (minClusters < 2) {
    throw new Error('minClusters must be at least 2');
  }
  if (maxClusters < minClusters) {
    throw new Error('maxClusters must be greater than or equal to minClusters');
  }

  // Convert data to tensor if needed
  const isInputTensor = X instanceof tf.Tensor;
  const dataTensor = isInputTensor ? X : tf.tensor2d(X as number[][]);
  const nSamples = dataTensor.shape[0];

  // Adjust maxClusters if it exceeds number of samples
  const effectiveMaxClusters = Math.min(maxClusters, nSamples - 1);
  
  if (effectiveMaxClusters < minClusters) {
    // Clean up tensor if we created it
    if (!isInputTensor) {
      dataTensor.dispose();
    }
    throw new Error(`Not enough samples (${nSamples}) for minimum clusters (${minClusters})`);
  }
  
  const evaluations: ClusterEvaluation[] = [];

  // Test each k value
  for (let k = minClusters; k <= effectiveMaxClusters; k++) {
      // Create clustering instance
      let clusterer;
      switch (algorithm) {
        case 'kmeans':
          clusterer = new KMeans({ nClusters: k, ...algorithmParams });
          break;
        case 'spectral':
          clusterer = new SpectralClustering({ nClusters: k, ...algorithmParams });
          break;
        case 'agglomerative':
          clusterer = new AgglomerativeClustering({ nClusters: k, ...algorithmParams });
          break;
        default:
          throw new Error(`Unknown algorithm: ${algorithm}`);
      }

      // Fit and predict
      const labelsTensor = await clusterer.fitPredict(dataTensor);
      
      // Convert labels to array if it's a tensor
      const labels = labelsTensor instanceof tf.Tensor 
        ? await labelsTensor.data() as unknown as number[]
        : labelsTensor;

      // Calculate metrics
      let silhouette = 0;
      let daviesBouldin = Infinity;
      let calinskiHarabasz = 0;

      if (metrics.includes('silhouette')) {
        silhouette = silhouetteScore(dataTensor, labels);
      }
      if (metrics.includes('daviesBouldin')) {
        daviesBouldin = daviesBouldinEfficient(dataTensor, labels);
      }
      if (metrics.includes('calinskiHarabasz')) {
        calinskiHarabasz = calinskiHarabaszEfficient(dataTensor, labels);
      }

      // Calculate combined score
      const evaluation: ClusterEvaluation = {
        k,
        silhouette,
        daviesBouldin,
        calinskiHarabasz,
        combinedScore: 0,
        labels: Array.from(labels)
      };
      
      // Dispose labels tensor if needed
      if (labelsTensor instanceof tf.Tensor) {
        labelsTensor.dispose();
      }

      // Use custom scoring function or default
      if (scoringFunction) {
        evaluation.combinedScore = scoringFunction(evaluation);
      } else {
        // Default: higher silhouette and calinski, lower davies = better
        evaluation.combinedScore = silhouette + calinskiHarabasz - daviesBouldin;
      }

      evaluations.push(evaluation);
    }

  // Sort by combined score (descending)
  evaluations.sort((a, b) => b.combinedScore - a.combinedScore);

  // Store result before cleanup
  const result = {
    optimal: evaluations[0],
    evaluations
  };

  // Clean up tensor if we created it
  if (!isInputTensor) {
    dataTensor.dispose();
  }

  return result;
}