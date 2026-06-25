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

export interface ClusterEvaluation {
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

export interface FindOptimalClustersOptions {
  /** Minimum number of clusters to test (default: 2) */
  min_clusters?: number;
  /** Maximum number of clusters to test (default: 10) */
  max_clusters?: number;
  /** Algorithm to use (default: 'kmeans') */
  algorithm?: 'kmeans' | 'spectral' | 'agglomerative' | 'som';
  /**
   * Algorithm-specific parameters forwarded to the clusterer constructor.
   * For `algorithm: 'agglomerative'`, `distance_threshold` is rejected because
   * `find_optimal_clusters` controls the stopping criterion via the k-sweep loop;
   * use `min_clusters`/`max_clusters` to bound the sweep instead.
   */
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

function normalize_and_score_evaluations(
  evaluations: ClusterEvaluation[],
  metrics: Array<'silhouette' | 'davies_bouldin' | 'calinski_harabasz'>,
): void {
  if (evaluations.length === 0) return;

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
  optimal: ClusterEvaluation;
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

  if (min_clusters < 2) {
    throw new Error('min_clusters must be at least 2');
  }
  if (max_clusters < min_clusters) {
    throw new Error('max_clusters must be greater than or equal to min_clusters');
  }
  if (algorithm === 'agglomerative' && 'distance_threshold' in algorithm_params) {
    throw new Error(
      "algorithm_params must not include 'distance_threshold': find_optimal_clusters controls the stopping criterion via the k-sweep loop.",
    );
  }

  const is_input_tensor = is_tensor(X);
  const data_tensor = is_input_tensor ? X : tf.tensor2d(X as number[][]);
  const n_samples = data_tensor.shape[0];

  const effective_max_clusters = Math.min(max_clusters, n_samples - 1);

  if (effective_max_clusters < min_clusters) {
    if (!is_input_tensor) {
      data_tensor.dispose();
    }
    throw new Error(
      `Not enough samples (${n_samples}) for minimum clusters (${min_clusters})`,
    );
  }

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

  // SOM training is independent of k: a single map is trained once, then each k
  // is produced by two-phase clustering (agglomerative grouping of the trained
  // neuron weight vectors into exactly k macro-clusters). This is the only
  // correct way to sweep k for SOM — the grid size does not equal the cluster
  // count. The grid is sized so the neuron count comfortably exceeds the
  // largest k while following the common ~5·√n heuristic for total neurons.
  let shared_som: SOM | null = null;
  if (algorithm === 'som') {
    const heuristic_grid = Math.ceil(Math.sqrt(5 * Math.sqrt(n_samples)));
    const min_grid = Math.ceil(Math.sqrt(effective_max_clusters));
    const grid_size = Math.max(2, heuristic_grid, min_grid);
    shared_som = new SOM({
      grid_width: grid_size,
      grid_height: grid_size,
      // algorithm_params may override grid dimensions and other SOM settings.
      ...algorithm_params,
    });
    await shared_som.fit(data_tensor);
  }

  for (let k = min_clusters; k <= effective_max_clusters; k++) {
    let labels: number[];
    let kmeans_instance: KMeans | null = null;
    let disposable:
      | KMeans
      | SpectralClustering
      | AgglomerativeClustering
      | null = null;

    if (algorithm === 'som') {
      // Phase 2: group trained neurons into exactly k macro-clusters and map
      // each sample (via its BMU) to a macro-cluster label.
      labels = await shared_som!.cluster(k);
    } else {
      let clusterer: KMeans | SpectralClustering | AgglomerativeClustering;
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
        default:
          throw new Error(`Unknown algorithm: ${algorithm}`);
      }

      labels = await clusterer.fit_predict(data_tensor);
      disposable = clusterer;
    }

    let silhouette = 0;
    let davies_bouldin = Infinity;
    let calinski_harabasz = 0;
    let wss: number | undefined;

    // Validation metrics require at least 2 distinct clusters. The SOM
    // two-phase path can collapse to fewer than k labels when all sample BMUs
    // fall into a single neuron macro-cluster; in that degenerate case assign
    // worst-case metric values (so this k ranks last) instead of crashing.
    const has_enough_clusters = new Set(labels).size >= 2;

    if (compute_silhouette) {
      silhouette = has_enough_clusters
        ? silhouette_score(data_tensor, labels, 'euclidean')
        : -1;
    }
    if (compute_db) {
      davies_bouldin = has_enough_clusters
        ? davies_bouldin_efficient(data_tensor, labels, 'euclidean')
        : Infinity;
    }
    if (compute_ch) {
      calinski_harabasz = has_enough_clusters
        ? calinski_harabasz_efficient(data_tensor, labels)
        : 0;
    }
    if (should_compute_wss) {
      // Optimize: read inertia directly from KMeans instead of recomputing
      if (kmeans_instance && kmeans_instance.inertia_ !== null) {
        wss = kmeans_instance.inertia_;
      } else {
        wss = compute_wss(data_tensor, labels);
      }
    }

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

    // Dispose the per-k clustering instance to free held tensors. (SOM is
    // trained once and disposed after the loop.)
    if (
      disposable &&
      'dispose' in disposable &&
      typeof disposable.dispose === 'function'
    ) {
      disposable.dispose();
    }
  }

  shared_som?.dispose();

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

  evaluations.sort((a, b) => b.combined_score - a.combined_score);

  const final_result = {
    optimal: evaluations[0],
    evaluations,
  };

  if (!is_input_tensor) {
    data_tensor.dispose();
  }

  return final_result;
}
