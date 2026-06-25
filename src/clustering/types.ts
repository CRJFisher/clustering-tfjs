import * as tf from '../backend/adapter';

/**
 * Allowing both `tf.Tensor2D` and plain nested arrays keeps the public API
 * flexible while internal paths can use the tensor directly.
 */
export type DataMatrix = tf.Tensor2D | number[][];

export type LabelVector = tf.Tensor1D | number[];

export type ClusteringMetric = 'euclidean' | 'manhattan' | 'cosine';

/**
 * Base for algorithms that do not use an explicit cluster count (e.g. SOM,
 * HDBSCAN); `BaseClusteringParams` extends this with `n_clusters`.
 */
export interface CoreClusteringParams {
  /** Only applies when the algorithm involves randomness (e.g., k-means init). */
  random_state?: number;
}

export interface BaseClusteringParams extends CoreClusteringParams {
  /** Must be ≥ 1. */
  n_clusters: number;
}

export interface KMeansParams extends BaseClusteringParams {
  max_iter?: number;

  /** Relative tolerance on inertia change to declare convergence. */
  tol?: number;

  /**
   * Runs k-means++ `n_init` times and keeps the solution with lowest inertia.
   * Mirrors scikit-learn's `n_init`. Must be ≥ 1. Defaults to 10.
   */
  n_init?: number;

  /**
   * `'cosine'` runs spherical k-means: rows are L2-normalized onto the unit
   * sphere and distances are cosine distances. `'euclidean'` (default) runs
   * standard Lloyd.
   */
  metric?: 'euclidean' | 'cosine';
}

export interface SpectralClusteringParams extends BaseClusteringParams {
  /**
   * Pre-defined strings or a callable returning a custom affinity matrix.
   * `'precomputed'` passes an `(n, n)` similarity matrix directly to `fit`.
   */
  affinity?:
    | 'rbf'
    | 'nearest_neighbors'
    | 'cosine'
    | 'precomputed'
    | ((X: DataMatrix) => tf.Tensor2D);

  /** Scaling parameter for the RBF kernel. */
  gamma?: number;

  /** Only used when `affinity` is `'nearest_neighbors'`. */
  n_neighbors?: number;

  /**
   * Mirrors scikit-learn's `n_init`. Defaults to **10**, which yields
   * considerably more robust assignments on challenging spectra than a single
   * initialisation.
   */
  n_init?: number;

  /**
   * Tries multiple k-means initialisations and selects the best by
   * Calinski-Harabász score. Particularly useful for 3+ cluster problems.
   */
  use_validation?: boolean;

  /** Number of random seeds tried when `use_validation` is enabled. Default: 20. */
  validation_attempts?: number;

  /**
   * - `'calinski-harabasz'`: Fast O(n·k), higher is better (default)
   * - `'davies-bouldin'`: O(n·k + k²), lower is better
   * - `'silhouette'`: Most accurate but O(n²), higher is better
   */
  validation_metric?: 'calinski-harabasz' | 'davies-bouldin' | 'silhouette';

  /**
   * Grid search over gamma values (RBF), validation attempts, and all metrics.
   * WARNING: computationally expensive; only suitable for small datasets.
   */
  intensive_parameter_sweep?: boolean;

  /** Only used when `intensive_parameter_sweep` is true and `affinity` is `'rbf'`. */
  gamma_range?: number[];

  /** Stores intermediate stats (affinity, Laplacian spectrum, embedding, metrics) accessible via `get_debug_info()`. */
  capture_debug_info?: boolean;

  /**
   * Spectral clustering requires O(n²) memory for the affinity matrix; raise
   * this only if you have sufficient memory. Default: 10 000.
   */
  max_samples?: number;
}

export interface HDBSCANParams extends CoreClusteringParams {
  /** Groups smaller than this become noise. Default 5, minimum 2. */
  min_cluster_size?: number;

  /**
   * Core-distance neighbourhood size. Defaults to `min_cluster_size`. The
   * point itself counts as its own first neighbour. Minimum 1.
   */
  min_samples?: number;

  /**
   * `'euclidean'` and `'manhattan'` are computed natively from the data; with
   * `'precomputed'`, `fit` accepts an `(n, n)` distance matrix directly (the
   * cosine path supplies a precomputed cosine distance matrix). Default
   * `'euclidean'`.
   */
  metric?: 'euclidean' | 'manhattan' | 'precomputed';

  /**
   * Flat-cut distance threshold: clusters whose birth distance
   * (`1 / birth_lambda`) is below `epsilon` are merged into a coarser ancestor.
   * Default 0 (disabled).
   */
  cluster_selection_epsilon?: number;

  /** Cluster extraction strategy. Default `'eom'` (Excess of Mass). */
  cluster_selection_method?: 'eom' | 'leaf';

  /** Store the most-persistent exemplar point per cluster. Default false. */
  store_exemplars?: boolean;
}

export interface AgglomerativeClusteringParams extends CoreClusteringParams {
  /**
   * Mutually exclusive with `distance_threshold`: provide exactly one.
   * Must be ≥ 1.
   */
  n_clusters?: number;

  /**
   * Clusters are not merged once their linkage distance reaches this threshold.
   * Mutually exclusive with `n_clusters`: provide exactly one.
   */
  distance_threshold?: number;

  linkage?: 'ward' | 'complete' | 'average' | 'single';

  /**
   * `'precomputed'` accepts a square, symmetric distance matrix (zero diagonal)
   * instead of a data matrix. Incompatible with `linkage: 'ward'`.
   */
  metric?: 'euclidean' | 'manhattan' | 'cosine' | 'precomputed';
}

export interface BaseClustering<
  Params extends CoreClusteringParams = BaseClusteringParams,
> {
  readonly params: Params;

  /** Async because underlying operations (GPU kernels, web workers) may not complete synchronously. */
  fit(X: DataMatrix): Promise<void>;

  fit_predict(X: DataMatrix): Promise<number[]>;
}

export type SOMTopology = 'rectangular' | 'hexagonal';

export type SOMNeighborhood = 'gaussian' | 'bubble' | 'mexican_hat';

export type SOMInitialization = 'random' | 'linear' | 'pca';

export type DecayFunction = (epoch: number, total_epochs: number) => number;

/**
 * SOM uses `grid_width × grid_height` neurons rather than `n_clusters`;
 * effective cluster count is determined after training via agglomerative
 * clustering on the neuron weight vectors.
 */
export interface SOMParams extends CoreClusteringParams {
  /** Must be ≥ 1. */
  grid_width: number;

  /** Must be ≥ 1. */
  grid_height: number;

  /**
   * - `'rectangular'`: 4 or 8 neighbours per neuron
   * - `'hexagonal'`: 6 neighbours per neuron
   */
  topology?: SOMTopology;

  /**
   * - `'gaussian'`: smooth exponential decay
   * - `'bubble'`: hard cutoff at radius
   * - `'mexican_hat'`: lateral inhibition pattern
   */
  neighborhood?: SOMNeighborhood;

  num_epochs?: number;

  /**
   * If a number: exponential decay from this initial value.
   * If a function: custom schedule `(epoch, total_epochs) => rate`.
   * Default: 0.5
   */
  learning_rate?: number | DecayFunction;

  /**
   * If a number: exponential decay from this initial value.
   * If a function: custom schedule `(epoch, total_epochs) => radius`.
   * Default: `max(grid_width, grid_height) / 2`
   */
  radius?: number | DecayFunction;

  initialization?: SOMInitialization;

  online_mode?: boolean;

  /** Only used when `online_mode` is true. Default: 32. */
  mini_batch_size?: number;

  /** Training stops when quantization error change < tol. Default: 1e-4. */
  tol?: number;

  /**
   * Explicit initial weight grid of shape `[grid_height][grid_width][n_features]`.
   * When provided, `fit()` and `partial_fit()` use these weights verbatim instead of
   * generating them from `initialization`/`random_state`, enabling reproducible
   * training and warm-starting from a known codebook. The feature dimension must
   * match the training data; a mismatch throws at fit time.
   */
  initial_weights?: number[][][];
}

export interface SOMState {
  weights: number[][][];
  total_samples: number;
  current_epoch: number;
  grid_width: number;
  grid_height: number;
  params: SOMParams;
}

export interface SOMClusterOptions {
  linkage?: 'ward' | 'complete' | 'average' | 'single';
  metric?: 'euclidean' | 'manhattan' | 'cosine';
}

export interface SOMMetrics {
  /** Average distance between samples and their Best Matching Units (BMUs). */
  quantization_error: number;

  /** Proportion of samples whose BMU and second-BMU are not topographic neighbours. */
  topographic_error: number;
}
