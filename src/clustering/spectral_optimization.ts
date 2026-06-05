import * as tf from '../backend/adapter';
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
  use_validation: boolean;
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
export async function validation_based_optimization(
  embedding: tf.Tensor2D,
  n_clusters: number,
  metric: 'calinski-harabasz' | 'davies-bouldin' | 'silhouette',
  attempts: number,
  random_state?: number,
): Promise<OptimizationResult> {
  const validation_module = await import('../validation');

  let best_labels: number[] | null = null;
  let best_score = metric === 'davies-bouldin' ? Infinity : -Infinity;

  // Try multiple random seeds
  for (let attempt = 0; attempt < attempts; attempt++) {
    const km_params = {
      n_clusters,
      random_state:
        random_state !== undefined ? random_state + attempt : undefined,
      n_init: 1, // Single run per seed when using validation
    };

    const km = new KMeans(km_params);
    await km.fit(embedding);
    const labels = km.labels_!;
    km.dispose();

    // Compute validation score based on selected metric
    let score: number;
    switch (metric) {
      case 'calinski-harabasz':
        score = validation_module.calinski_harabasz(embedding, labels);
        break;
      case 'davies-bouldin':
        score = validation_module.davies_bouldin(embedding, labels);
        break;
      case 'silhouette':
        score = validation_module.silhouette_score(embedding, labels);
        break;
    }

    // Update best score (lower is better for Davies-Bouldin)
    const is_better =
      metric === 'davies-bouldin' ? score < best_score : score > best_score;

    if (is_better) {
      best_score = score;
      best_labels = labels;
    }
  }

  return {
    labels: best_labels!,
    config: {
      gamma: 0, // Will be set by caller
      metric,
      attempts,
      use_validation: true,
    },
    score: best_score,
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
export async function intensive_parameter_sweep(
  X: tf.Tensor2D,
  params: SpectralClusteringParams,
  compute_embedding_from_affinity: (
    affinity_matrix: tf.Tensor2D,
  ) => Promise<tf.Tensor2D>,
  compute_affinity_matrix: (
    X: tf.Tensor2D,
    params: SpectralClusteringParams,
  ) => tf.Tensor2D,
): Promise<OptimizationResult> {
  const validation_module = await import('../validation');
  const gamma_range = params.gamma_range ?? [
    0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0,
  ];
  const metrics: Array<'calinski-harabasz' | 'davies-bouldin' | 'silhouette'> =
    ['calinski-harabasz', 'davies-bouldin', 'silhouette'];
  const attempts_range = [10, 20, 30];

  let best_result: OptimizationResult = {
    labels: [],
    config: {
      gamma: params.gamma ?? 1.0,
      metric: 'calinski-harabasz',
      attempts: 20,
      use_validation: false,
    },
  };
  let best_score = -Infinity;

  // Process each gamma value: compute affinity + embedding ONCE,
  // then run all k-means variations against the cached embedding.
  for (const gamma of gamma_range) {
    const affinity_matrix = compute_affinity_matrix(X, {
      ...params,
      gamma,
    });

    const embedding = await compute_embedding_from_affinity(affinity_matrix);
    affinity_matrix.dispose(); // No longer needed after embedding is computed

    try {
      // Phase A: non-validation k-means with n_init=10
      const km = new KMeans({
        n_clusters: params.n_clusters,
        random_state: params.random_state,
        n_init: 10,
      });

      await km.fit(embedding);
      const labels = km.labels_!;
      km.dispose();

      // Evaluate with all metrics
      let avg_score = 0;
      for (const metric of metrics) {
        let score: number;
        switch (metric) {
          case 'calinski-harabasz':
            score = validation_module.calinski_harabasz(embedding, labels);
            break;
          case 'davies-bouldin':
            score = -validation_module.davies_bouldin(embedding, labels);
            break;
          case 'silhouette':
            score = validation_module.silhouette_score(embedding, labels);
            break;
        }
        avg_score += score;
      }
      avg_score /= metrics.length;

      if (avg_score > best_score) {
        best_score = avg_score;
        best_result = {
          labels,
          config: {
            gamma,
            metric: 'calinski-harabasz',
            attempts: 0,
            use_validation: false,
          },
        };
      }

      // Phase B: validation-based optimization (reusing the same embedding)
      for (const attempts of attempts_range) {
        for (const metric of metrics) {
          try {
            const result = await validation_based_optimization(
              embedding,
              params.n_clusters,
              metric,
              attempts,
              params.random_state,
            );

            const normalized_score =
              metric === 'davies-bouldin'
                ? -(result.score ?? 0)
                : (result.score ?? 0);

            if (normalized_score > best_score) {
              best_score = normalized_score;
              best_result = {
                labels: result.labels,
                config: {
                  gamma,
                  metric,
                  attempts,
                  use_validation: true,
                },
              };
            }
          } catch {
            // Skip if validation fails
          }
        }
      }
    } finally {
      embedding.dispose(); // Guaranteed cleanup
    }
  }

  return best_result;
}
