import * as tf from '../tf-adapter';
import type {
  SpectralClusteringParams,
  DataMatrix as _DataMatrix,
} from './types';
import { KMeans } from './kmeans';

/**
 * Configuration for parameter optimization
 */
interface OptimizationConfig {
  gamma: number;
  metric: 'calinski-harabasz' | 'davies-bouldin' | 'silhouette';
  attempts: number;
  useValidation: boolean;
}

/**
 * Result of optimization
 */
interface OptimizationResult {
  labels: number[];
  config: OptimizationConfig;
  score?: number;
}

/**
 * Performs validation-based optimization for spectral clustering.
 * Tries multiple k-means initializations and selects the best based on validation score.
 */
export async function validationBasedOptimization(
  embedding: tf.Tensor2D,
  nClusters: number,
  metric: 'calinski-harabasz' | 'davies-bouldin' | 'silhouette',
  attempts: number,
  randomState?: number,
): Promise<OptimizationResult> {
  const validationModule = await import('../validation');

  let bestLabels: number[] | null = null;
  let bestScore = metric === 'davies-bouldin' ? Infinity : -Infinity;

  // Try multiple random seeds
  for (let attempt = 0; attempt < attempts; attempt++) {
    const kmParams = {
      nClusters,
      randomState:
        randomState !== undefined ? randomState + attempt : undefined,
      nInit: 1, // Single run per seed when using validation
    };

    const km = new KMeans(kmParams);
    await km.fit(embedding);
    const labels = km.labels_ as number[];
    km.dispose();

    // Compute validation score based on selected metric
    let score: number;
    switch (metric) {
      case 'calinski-harabasz':
        score = validationModule.calinskiHarabasz(embedding, labels);
        break;
      case 'davies-bouldin':
        score = validationModule.daviesBouldin(embedding, labels);
        break;
      case 'silhouette':
        score = validationModule.silhouetteScore(embedding, labels);
        break;
    }

    // Update best score (lower is better for Davies-Bouldin)
    const isBetter =
      metric === 'davies-bouldin' ? score < bestScore : score > bestScore;

    if (isBetter) {
      bestScore = score;
      bestLabels = labels;
    }
  }

  return {
    labels: bestLabels!,
    config: {
      gamma: 0, // Will be set by caller
      metric,
      attempts,
      useValidation: true,
    },
    score: bestScore,
  };
}

/**
 * Performs intensive parameter sweep for difficult clustering problems.
 * Tests multiple gamma values and validation configurations.
 *
 * Optimized: computes the eigendecomposition (via embedding) only ONCE per
 * gamma value, then reuses it across all metric/attempt combinations.
 * This reduces eigendecompositions from 9×(1+3×3)=90 to just 9.
 */
export async function intensiveParameterSweep(
  X: tf.Tensor2D,
  params: SpectralClusteringParams,
  computeEmbeddingFromAffinity: (
    affinityMatrix: tf.Tensor2D,
  ) => Promise<tf.Tensor2D>,
  computeAffinityMatrix: (
    X: tf.Tensor2D,
    params: SpectralClusteringParams,
  ) => tf.Tensor2D,
): Promise<OptimizationResult> {
  const validationModule = await import('../validation');
  const gammaRange = params.gammaRange ?? [
    0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0,
  ];
  const metrics: Array<'calinski-harabasz' | 'davies-bouldin' | 'silhouette'> =
    ['calinski-harabasz', 'davies-bouldin', 'silhouette'];
  const attemptsRange = [10, 20, 30];

  let bestResult: OptimizationResult = {
    labels: [],
    config: {
      gamma: params.gamma ?? 1.0,
      metric: 'calinski-harabasz',
      attempts: 20,
      useValidation: false,
    },
  };
  let bestScore = -Infinity;

  // Process each gamma value: compute affinity + embedding ONCE,
  // then run all k-means variations against the cached embedding.
  for (const gamma of gammaRange) {
    const affinityMatrix = computeAffinityMatrix(X, {
      ...params,
      gamma,
    });

    const embedding = await computeEmbeddingFromAffinity(affinityMatrix);
    affinityMatrix.dispose(); // No longer needed after embedding is computed

    try {
      // Phase A: non-validation k-means with nInit=10
      const km = new KMeans({
        nClusters: params.nClusters,
        randomState: params.randomState,
        nInit: 10,
      });

      await km.fit(embedding);
      const labels = km.labels_ as number[];
      km.dispose();

      // Evaluate with all metrics
      let avgScore = 0;
      for (const metric of metrics) {
        let score: number;
        switch (metric) {
          case 'calinski-harabasz':
            score = validationModule.calinskiHarabasz(embedding, labels);
            break;
          case 'davies-bouldin':
            score = -validationModule.daviesBouldin(embedding, labels);
            break;
          case 'silhouette':
            score = validationModule.silhouetteScore(embedding, labels);
            break;
        }
        avgScore += score;
      }
      avgScore /= metrics.length;

      if (avgScore > bestScore) {
        bestScore = avgScore;
        bestResult = {
          labels,
          config: {
            gamma,
            metric: 'calinski-harabasz',
            attempts: 0,
            useValidation: false,
          },
        };
      }

      // Phase B: validation-based optimization (reusing the same embedding)
      for (const attempts of attemptsRange) {
        for (const metric of metrics) {
          try {
            const result = await validationBasedOptimization(
              embedding,
              params.nClusters,
              metric,
              attempts,
              params.randomState,
            );

            const normalizedScore =
              metric === 'davies-bouldin'
                ? -(result.score ?? 0)
                : (result.score ?? 0);

            if (normalizedScore > bestScore) {
              bestScore = normalizedScore;
              bestResult = {
                labels: result.labels,
                config: {
                  gamma,
                  metric,
                  attempts,
                  useValidation: true,
                },
              };
            }
          } catch (_e) {
            // Skip if validation fails
          }
        }
      }
    } finally {
      embedding.dispose(); // Guaranteed cleanup
    }
  }

  return bestResult;
}
