import * as tf from '../backend/adapter';
import { KMeans } from '../clustering/kmeans';
import { SpectralClustering } from '../clustering/spectral';
import { AgglomerativeClustering } from '../clustering/agglomerative';
import { SOM } from '../clustering/som';
import { silhouette_score } from '../validation/silhouette';
import { davies_bouldin_efficient } from '../validation/davies_bouldin';
import { calinski_harabasz_efficient } from '../validation/calinski_harabasz';
import { is_tensor } from '../tensor/tensor_guards';
import { compute_wss } from './compute_wss';
import { find_knee } from './kneedle';
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
  davies_bouldin: number;
  /** Calinski-Harabasz index (range: [0, ∞), higher is better) */
  calinski_harabasz: number;
  /**
   * Combined score used for selection.
   * When using the default 'combined' method, this is the mean of normalized
   * metrics in [0, 1]. When using 'silhouette', this is the raw silhouette.
   * When using 'elbow', the knee point gets score 1.0.
   */
  combined_score: number;
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
  min_clusters?: number;
  /** Maximum number of clusters to test (default: 10) */
  max_clusters?: number;
  /** Algorithm to use (default: 'kmeans') */
  algorithm?: 'kmeans' | 'spectral' | 'agglomerative' | 'som';
  /** Algorithm-specific parameters */
  algorithm_params?: Record<string, unknown>;
  /** Metrics to use for evaluation (default: all). Only used with 'combined' method. */
  metrics?: Array<'silhouette' | 'davies_bouldin' | 'calinski_harabasz'>;
  /**
   * Custom scoring function. Receives raw (un-normalized) metric values.
   * Overrides the `method` option when provided.
   */
  scoring_function?: (evaluation: ClusterEvaluation) => number;
  /** Method for selecting optimal k (default: 'combined') */
  method?: OptimalClustersMethod;
}

/**
 * Normalizes metrics across all evaluations and computes combined scores.
 * Each metric is normalized to [0, 1] and averaged for equal contribution.
 */
function normalize_and_score_evaluations(
  evaluations: ClusterEvaluation[],
  metrics: Array<'silhouette' | 'davies_bouldin' | 'calinski_harabasz'>,
): void {
  if (evaluations.length === 0) return;

  // Pre-compute min-max for Calinski-Harabasz (higher is better)
  let ch_min = Infinity;
  let ch_max = -Infinity;
  if (metrics.includes('calinski_harabasz')) {
    for (const e of evaluations) {
      if (Number.isFinite(e.calinski_harabasz)) {
        ch_min = Math.min(ch_min, e.calinski_harabasz);
        ch_max = Math.max(ch_max, e.calinski_harabasz);
      }
    }
  }
  const ch_range = ch_max - ch_min;

  // Pre-compute min-max for Davies-Bouldin (lower is better)
  let db_min = Infinity;
  let db_max = -Infinity;
  if (metrics.includes('davies_bouldin')) {
    for (const e of evaluations) {
      if (Number.isFinite(e.davies_bouldin)) {
        db_min = Math.min(db_min, e.davies_bouldin);
        db_max = Math.max(db_max, e.davies_bouldin);
      }
    }
  }
  const db_range = db_max - db_min;

  // Score each evaluation with normalized metrics
  for (const evaluation of evaluations) {
    let score = 0;
    let num_metrics = 0;

    if (metrics.includes('silhouette')) {
      // Fixed range normalization: [-1, 1] -> [0, 1]
      const ns = (evaluation.silhouette + 1) / 2;
      score += Number.isFinite(ns) ? ns : 0;
      num_metrics++;
    }

    if (metrics.includes('calinski_harabasz')) {
      let nch: number;
      if (!Number.isFinite(evaluation.calinski_harabasz)) {
        // CH = Infinity means perfect separation (WSS = 0) -> best score
        nch = 1;
      } else if (ch_range === 0 || !Number.isFinite(ch_range)) {
        nch = 0.5;
      } else {
        nch = (evaluation.calinski_harabasz - ch_min) / ch_range;
      }
      score += nch;
      num_metrics++;
    }

    if (metrics.includes('davies_bouldin')) {
      let ndb: number;
      if (!Number.isFinite(evaluation.davies_bouldin)) {
        ndb = 0;
      } else if (db_range === 0 || !Number.isFinite(db_range)) {
        ndb = 0.5;
      } else {
        ndb = 1 - (evaluation.davies_bouldin - db_min) / db_range;
      }
      score += ndb;
      num_metrics++;
    }

    evaluation.combined_score = num_metrics > 0 ? score / num_metrics : 0;
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
 * import { find_optimal_clusters } from 'clustering-tfjs';
 *
 * const data = [[1, 2], [1.5, 1.8], [5, 8], [8, 8], [1, 0.6], [9, 11]];
 * const result = await find_optimal_clusters(data, { max_clusters: 5 });
 *
 * console.log(`Optimal number of clusters: ${result.optimal.k}`);
 * console.log(`Best silhouette score: ${result.optimal.silhouette}`);
 * ```
 */
export async function find_optimal_clusters(
  X: DataMatrix,
  options: FindOptimalClustersOptions = {},
): Promise<{
  /** The optimal cluster evaluation */
  optimal: ClusterEvaluation;
  /** All evaluations sorted by combined score (descending) */
  evaluations: ClusterEvaluation[];
}> {
  const {
    min_clusters = 2,
    max_clusters = 10,
    algorithm = 'kmeans',
    algorithm_params = {},
    metrics = ['silhouette', 'davies_bouldin', 'calinski_harabasz'],
    scoring_function,
    method = 'combined',
  } = options;

  // Validate inputs
  if (min_clusters < 2) {
    throw new Error('min_clusters must be at least 2');
  }
  if (max_clusters < min_clusters) {
    throw new Error('max_clusters must be greater than or equal to min_clusters');
  }

  // Convert data to tensor if needed
  const is_input_tensor = is_tensor(X);
  const data_tensor = is_input_tensor ? X : tf.tensor2d(X as number[][]);
  const n_samples = data_tensor.shape[0];

  // Adjust max_clusters if it exceeds number of samples
  const effective_max_clusters = Math.min(max_clusters, n_samples - 1);

  if (effective_max_clusters < min_clusters) {
    // Clean up tensor if we created it
    if (!is_input_tensor) {
      data_tensor.dispose();
    }
    throw new Error(
      `Not enough samples (${n_samples}) for minimum clusters (${min_clusters})`,
    );
  }

  // Determine which metrics to compute based on method
  const compute_silhouette =
    method === 'silhouette' ||
    (method === 'combined' && metrics.includes('silhouette')) ||
    (!!scoring_function && metrics.includes('silhouette'));
  const compute_db =
    (method === 'combined' && metrics.includes('davies_bouldin')) ||
    (!!scoring_function && metrics.includes('davies_bouldin'));
  const compute_ch =
    (method === 'combined' && metrics.includes('calinski_harabasz')) ||
    (!!scoring_function && metrics.includes('calinski_harabasz'));
  const should_compute_wss = method === 'elbow';

  const evaluations: ClusterEvaluation[] = [];

  // Test each k value
  for (let k = min_clusters; k <= effective_max_clusters; k++) {
    // Create clustering instance
    let clusterer;
    let kmeans_instance: KMeans | null = null;

    switch (algorithm) {
      case 'kmeans':
        kmeans_instance = new KMeans({ n_clusters: k, ...algorithm_params });
        clusterer = kmeans_instance;
        break;
      case 'spectral':
        clusterer = new SpectralClustering({
          n_clusters: k,
          ...algorithm_params,
        });
        break;
      case 'agglomerative':
        clusterer = new AgglomerativeClustering({
          n_clusters: k,
          ...algorithm_params,
        });
        break;
      case 'som': {
        // For SOM, we need to determine grid dimensions
        // Use square grid as default, can be overridden via algorithm_params
        const grid_size = Math.ceil(Math.sqrt(k));
        const params = algorithm_params as Record<string, unknown>;
        clusterer = new SOM({
          grid_width: (params.grid_width as number) || grid_size,
          grid_height: (params.grid_height as number) || Math.ceil(k / grid_size),
          ...algorithm_params,
        });
        break;
      }
      default:
        throw new Error(`Unknown algorithm: ${algorithm}`);
    }

    // Fit and predict
    const labels = await clusterer.fit_predict(data_tensor);

    // Calculate metrics
    let silhouette = 0;
    let davies_bouldin = Infinity;
    let calinski_harabasz = 0;
    let wss: number | undefined;

    if (compute_silhouette) {
      silhouette = silhouette_score(data_tensor, labels);
    }
    if (compute_db) {
      davies_bouldin = davies_bouldin_efficient(data_tensor, labels);
    }
    if (compute_ch) {
      calinski_harabasz = calinski_harabasz_efficient(data_tensor, labels);
    }
    if (should_compute_wss) {
      // Optimize: read inertia directly from KMeans instead of recomputing
      if (kmeans_instance && kmeans_instance.inertia_ !== null) {
        wss = kmeans_instance.inertia_;
      } else {
        wss = compute_wss(data_tensor, labels);
      }
    }

    // Build evaluation
    const evaluation: ClusterEvaluation = {
      k,
      silhouette,
      davies_bouldin,
      calinski_harabasz,
      combined_score: 0,
      labels: Array.from(labels),
    };
    if (wss !== undefined) {
      evaluation.wss = wss;
    }

    // Apply custom scoring function immediately (gets raw values)
    if (scoring_function) {
      evaluation.combined_score = scoring_function(evaluation);
    }

    evaluations.push(evaluation);

    // Dispose clustering instance to free held tensors
    const disposable = clusterer as { dispose?: () => void };
    if (typeof disposable.dispose === 'function') {
      disposable.dispose();
    }
  }

  // Post-loop scoring based on method (only when no custom scorer)
  if (!scoring_function) {
    if (method === 'combined') {
      normalize_and_score_evaluations(evaluations, metrics);
    } else if (method === 'silhouette') {
      for (const evaluation of evaluations) {
        evaluation.combined_score = evaluation.silhouette;
      }
    } else if (method === 'elbow') {
      const k_values = evaluations.map((e) => e.k);
      const wss_values = evaluations.map((e) => e.wss!);
      const result = find_knee(k_values, wss_values, { direction: 'concave' });

      if (result.knee_x !== null) {
        // For concave curves, differences are negative (curve below diagonal).
        // Negate so the knee (most negative diff) gets the highest score.
        for (let i = 0; i < evaluations.length; i++) {
          evaluations[i].combined_score = -result.differences[i];
        }
      } else {
        // Fallback: prefer smallest k (parsimony)
        for (const evaluation of evaluations) {
          evaluation.combined_score = -evaluation.k;
        }
      }
    }
  }

  // Sort by combined score (descending)
  evaluations.sort((a, b) => b.combined_score - a.combined_score);

  // Store result before cleanup
  const final_result = {
    optimal: evaluations[0],
    evaluations,
  };

  // Clean up tensor if we created it
  if (!is_input_tensor) {
    data_tensor.dispose();
  }

  return final_result;
}
