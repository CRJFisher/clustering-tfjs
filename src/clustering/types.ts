import * as tf from '../backend/adapter';

/**
 * ----------------------------------------------------------------------------
 * Generic type aliases used across clustering algorithms
 * ----------------------------------------------------------------------------
 */

/**
 * Two-dimensional data matrix accepted by `fit`/`fit_predict` methods.
 *
 * Either a `tf.Tensor2D` **or** a plain nested JavaScript array. Allowing both
 * keeps the public API flexible while still supporting efficient tensor based
 * back-ends internally.
 */
export type DataMatrix = tf.Tensor2D | number[][];

/**
 * One-dimensional vector of cluster assignments returned by `fit_predict`.
 *
 * The choice of `tf.Tensor1D` or plain `number[]` again mirrors `DataMatrix`.
 */
export type LabelVector = tf.Tensor1D | number[];

/**
 * Distance metric shared across estimators and representation accessors.
 */
export type ClusteringMetric = 'euclidean' | 'manhattan' | 'cosine';

/**
 * Minimal options shared by *all* clustering algorithms, including those
 * that do not use an explicit cluster count (e.g. SOM).
 */
export interface CoreClusteringParams {
  /**
   * Optional random seed to ensure deterministic behaviour where the
   * underlying algorithm involves randomness (e.g. k-means init).
   */
  random_state?: number;
}

/**
 * Common options for clustering algorithms that require an explicit cluster count.
 */
export interface BaseClusteringParams extends CoreClusteringParams {
  /**
   * Desired number of clusters. Must be ≥ 1.
   */
  n_clusters: number;
}

/**
 * ----------------------------------------------------------------------------
 * Algorithm specific parameter objects
 * ----------------------------------------------------------------------------
 *
 * Only parameters required during *construction* or the first call to `fit`
 * should be included here. Runtime options specific to a single method call
 * can be expressed via regular function parameters (future tasks).
 */

export interface KMeansParams extends BaseClusteringParams {
  /**
   * Maximum number of iterations for the Lloyd update phase.
   */
  max_iter?: number;

  /**
   * Relative tolerance with regards to inertia to declare convergence.
   */
  tol?: number;

  /**
   * Number of random initialisations to perform. The algorithm will run
   * k-means++ `n_init` times and keep the solution with the lowest inertia.
   * Mirrors scikit-learn’s `n_init` parameter. Must be ≥ 1. Defaults to 10.
   */
  n_init?: number;

  /**
   * Distance metric. `'cosine'` runs spherical k-means: rows are L2-normalized
   * onto the unit sphere and all distances are cosine distances, routed through
   * `pairwise_distance_matrix`. `'euclidean'` (default) runs standard Lloyd.
   */
  metric?: 'euclidean' | 'cosine';
}

export interface SpectralClusteringParams extends BaseClusteringParams {
  /**
   * The affinity metric to build the similarity graph.
   * Either one of the pre-defined strings or a callable returning an
   * affinity matrix.
   */
  affinity?:
    | 'rbf'
    | 'nearest_neighbors'
    | 'cosine'
    | 'precomputed'
    | ((X: DataMatrix) => tf.Tensor2D);

  /**
   * Scaling parameter for the RBF kernel (if selected).
   */
  gamma?: number;

  /**
   * Number of nearest neighbours to connect in the k-NN similarity graph.
   * Only used when `affinity` is set to "nearest_neighbors".
   */
  n_neighbors?: number;

  /**
   * Number of random initialisations for the inner K-Means step. Mirrors
   * scikit-learn's `n_init` parameter. If omitted the algorithm defaults to
   * **10** which yields considerably more robust cluster assignments on
   * challenging spectra than a single initialisation.
   */
  n_init?: number;

  /**
   * Whether to use validation metrics to optimize clustering parameters.
   * When enabled, the algorithm will try multiple parameter combinations
   * and k-means initializations, selecting the best result based on
   * Calinski-Harabasz score. Particularly useful for 3+ cluster problems.
   */
  use_validation?: boolean;

  /**
   * Number of different random seeds to try when use_validation is enabled.
   * Each seed generates a different k-means++ initialization.
   * Default: 20
   */
  validation_attempts?: number;

  /**
   * Whether to also optimize affinity parameters (gamma for RBF, n_neighbors for kNN)
   * when use_validation is enabled. This performs a grid search over parameter values.
   * Default: false
   */
  optimize_affinity_params?: boolean;

  /**
   * Which validation metric to use for optimization when use_validation is enabled.
   * - 'calinski-harabasz': Fast O(n·k), higher is better (default)
   * - 'davies-bouldin': O(n·k + k²), lower is better
   * - 'silhouette': Most accurate but O(n²), higher is better
   * Default: 'calinski-harabasz'
   */
  validation_metric?: 'calinski-harabasz' | 'davies-bouldin' | 'silhouette';

  /**
   * Enable intensive parameter sweep for difficult clustering problems.
   * When enabled, performs grid search over:
   * - Multiple gamma values (for RBF affinity)
   * - Multiple validation attempts
   * - All validation metrics
   * WARNING: This is computationally expensive and should only be used
   * for small datasets or when accuracy is critical.
   * Default: false
   */
  intensive_parameter_sweep?: boolean;

  /**
   * Custom gamma values to test during intensive parameter sweep.
   * Only used when intensive_parameter_sweep is true and affinity is 'rbf'.
   * Default: [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
   */
  gamma_range?: number[];

  /**
   * Whether to capture debug information during fitting.
   * When enabled, intermediate statistics (affinity, Laplacian spectrum,
   * embedding, clustering metrics) are stored and accessible via get_debug_info().
   * Default: false
   */
  capture_debug_info?: boolean;

  /**
   * Maximum number of samples allowed. Spectral clustering requires O(n^2)
   * memory for the affinity matrix, so very large datasets can cause OOM.
   * Set to a higher value if you have sufficient memory.
   * Default: 10000
   */
  max_samples?: number;
}

export interface HDBSCANParams extends CoreClusteringParams {
  /**
   * Minimum cluster size; groups smaller than this become noise. Default 5,
   * minimum 2.
   */
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

export interface AgglomerativeClusteringParams extends BaseClusteringParams {
  /**
   * Linkage criterion used to merge clusters.
   */
  linkage?: 'ward' | 'complete' | 'average' | 'single';

  /**
   * Metric to compute linkage.
   */
  metric?: 'euclidean' | 'manhattan' | 'cosine';
}

/**
 * ----------------------------------------------------------------------------
 * Base interface to be implemented by *all* clustering estimators
 * ----------------------------------------------------------------------------
 */

export interface BaseClustering<
  Params extends CoreClusteringParams = BaseClusteringParams,
> {
  /**
   * Hyper-parameters used by the estimator instance.
   */
  readonly params: Params;

  /**
   * Fits the model to the provided data. Returns a `Promise` that resolves
   * when the underlying asynchronous operations (GPU kernels, web workers,
   * etc.) have finished.
   */
  fit(X: DataMatrix): Promise<void>;

  /**
   * Convenience wrapper that fits the model **and** immediately returns the
   * predicted labels for `X` in a single call.
   */
  fit_predict(X: DataMatrix): Promise<number[]>;
}

/**
 * ----------------------------------------------------------------------------
 * Self-Organizing Maps (SOM) specific types and interfaces
 * ----------------------------------------------------------------------------
 */

/**
 * Grid topology for the SOM neurons arrangement.
 */
export type SOMTopology = 'rectangular' | 'hexagonal';

/**
 * Neighborhood function for determining influence of BMU on surrounding neurons.
 */
export type SOMNeighborhood = 'gaussian' | 'bubble' | 'mexican_hat';

/**
 * Weight initialization strategy for the SOM grid.
 */
export type SOMInitialization = 'random' | 'linear' | 'pca';

/**
 * Decay function type for learning rate and radius scheduling.
 */
export type DecayFunction = (epoch: number, total_epochs: number) => number;

/**
 * Parameters for Self-Organizing Maps algorithm.
 *
 * SOM uses grid_width * grid_height to define the map size rather than n_clusters.
 */
export interface SOMParams extends CoreClusteringParams {
  /**
   * Width of the SOM grid (number of neurons in x-axis).
   * Must be ≥ 1.
   */
  grid_width: number;

  /**
   * Height of the SOM grid (number of neurons in y-axis).
   * Must be ≥ 1.
   */
  grid_height: number;

  /**
   * Grid topology determining neuron connectivity.
   * - 'rectangular': 4 or 8 neighbors per neuron
   * - 'hexagonal': 6 neighbors per neuron
   * Default: 'rectangular'
   */
  topology?: SOMTopology;

  /**
   * Neighborhood function for weight updates.
   * - 'gaussian': Smooth exponential decay
   * - 'bubble': Hard cutoff at radius
   * - 'mexican_hat': Lateral inhibition pattern
   * Default: 'gaussian'
   */
  neighborhood?: SOMNeighborhood;

  /**
   * Number of training epochs.
   * Default: 100
   */
  num_epochs?: number;

  /**
   * Initial learning rate or custom decay function.
   * If number: uses exponential decay from this initial value.
   * If function: custom decay schedule (epoch, total_epochs) => rate.
   * Default: 0.5
   */
  learning_rate?: number | DecayFunction;

  /**
   * Initial neighborhood radius or custom decay function.
   * If number: uses exponential decay from this initial value.
   * If function: custom decay schedule (epoch, total_epochs) => radius.
   * Default: max(grid_width, grid_height) / 2
   */
  radius?: number | DecayFunction;

  /**
   * Weight initialization strategy.
   * - 'random': Random values from input data range
   * - 'linear': Along first two principal components
   * - 'pca': Using PCA of input data
   * Default: 'random'
   */
  initialization?: SOMInitialization;

  /**
   * Enable online/incremental learning mode.
   * When true, allows using partial_fit() for streaming data.
   * Default: false
   */
  online_mode?: boolean;

  /**
   * Mini-batch size for online learning.
   * Only used when online_mode is true.
   * Default: 32
   */
  mini_batch_size?: number;

  /**
   * Convergence tolerance for early stopping.
   * Training stops when quantization error change < tol.
   * Default: 1e-4
   */
  tol?: number;
}

/**
 * State object for SOM persistence and online learning.
 */
export interface SOMState {
  /**
   * Current weight matrix [height, width, features].
   */
  weights: number[][][];

  /**
   * Total number of samples learned.
   */
  total_samples: number;

  /**
   * Current epoch number (for online learning).
   */
  current_epoch: number;

  /**
   * Grid dimensions.
   */
  grid_width: number;
  grid_height: number;

  /**
   * Configuration parameters.
   */
  params: SOMParams;
}

/**
 * Options for the SOM 2-phase cluster() method.
 */
export interface SOMClusterOptions {
  /**
   * Linkage criterion for agglomerative clustering on SOM neurons.
   * Default: 'ward'
   */
  linkage?: 'ward' | 'complete' | 'average' | 'single';

  /**
   * Distance metric for agglomerative clustering on SOM neurons.
   * Default: 'euclidean'
   */
  metric?: 'euclidean' | 'manhattan' | 'cosine';
}

/**
 * Metrics for evaluating SOM quality.
 */
export interface SOMMetrics {
  /**
   * Average distance between samples and their BMUs.
   */
  quantization_error: number;

  /**
   * Proportion of samples whose BMU and second BMU are not neighbors.
   */
  topographic_error: number;
}
