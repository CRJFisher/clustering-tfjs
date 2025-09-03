import * as tf from '../tf-adapter';

/**
 * ----------------------------------------------------------------------------
 * Generic type aliases used across clustering algorithms
 * ----------------------------------------------------------------------------
 */

/**
 * Two-dimensional data matrix accepted by `fit`/`fitPredict` methods.
 *
 * Either a `tf.Tensor2D` **or** a plain nested JavaScript array. Allowing both
 * keeps the public API flexible while still supporting efficient tensor based
 * back-ends internally.
 */
export type DataMatrix = tf.Tensor2D | number[][];

/**
 * One-dimensional vector of cluster assignments returned by `fitPredict`.
 *
 * The choice of `tf.Tensor1D` or plain `number[]` again mirrors `DataMatrix`.
 */
export type LabelVector = tf.Tensor1D | number[];

/**
 * Common options shared by *all* clustering algorithms.
 */
export interface BaseClusteringParams {
  /**
   * Desired number of clusters. Must be ≥ 1.
   */
  nClusters: number;

  /**
   * Optional random seed to ensure deterministic behaviour where the
   * underlying algorithm involves randomness (e.g. k-means init).
   */
  randomState?: number;
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
  maxIter?: number;

  /**
   * Relative tolerance with regards to inertia to declare convergence.
   */
  tol?: number;

  /**
   * Number of random initialisations to perform. The algorithm will run
   * k-means++ `nInit` times and keep the solution with the lowest inertia.
   * Mirrors scikit-learn’s `n_init` parameter. Must be ≥ 1. Defaults to 10.
   */
  nInit?: number;
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
  nNeighbors?: number;

  /**
   * Number of random initialisations for the inner K-Means step. Mirrors
   * scikit-learn's `n_init` parameter. If omitted the algorithm defaults to
   * **10** which yields considerably more robust cluster assignments on
   * challenging spectra than a single initialisation.
   */
  nInit?: number;

  /**
   * Whether to use validation metrics to optimize clustering parameters.
   * When enabled, the algorithm will try multiple parameter combinations
   * and k-means initializations, selecting the best result based on
   * Calinski-Harabasz score. Particularly useful for 3+ cluster problems.
   */
  useValidation?: boolean;

  /**
   * Number of different random seeds to try when useValidation is enabled.
   * Each seed generates a different k-means++ initialization.
   * Default: 20
   */
  validationAttempts?: number;

  /**
   * Whether to also optimize affinity parameters (gamma for RBF, nNeighbors for kNN)
   * when useValidation is enabled. This performs a grid search over parameter values.
   * Default: false
   */
  optimizeAffinityParams?: boolean;

  /**
   * Which validation metric to use for optimization when useValidation is enabled.
   * - 'calinski-harabasz': Fast O(n·k), higher is better (default)
   * - 'davies-bouldin': O(n·k + k²), lower is better
   * - 'silhouette': Most accurate but O(n²), higher is better
   * Default: 'calinski-harabasz'
   */
  validationMetric?: 'calinski-harabasz' | 'davies-bouldin' | 'silhouette';

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
  intensiveParameterSweep?: boolean;

  /**
   * Custom gamma values to test during intensive parameter sweep.
   * Only used when intensiveParameterSweep is true and affinity is 'rbf'.
   * Default: [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
   */
  gammaRange?: number[];
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
  Params extends BaseClusteringParams = BaseClusteringParams,
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
  fitPredict(X: DataMatrix): Promise<LabelVector>;
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
export type DecayFunction = (epoch: number, totalEpochs: number) => number;

/**
 * Parameters for Self-Organizing Maps algorithm.
 * 
 * Note: SOM doesn't use `nClusters` from BaseClusteringParams.
 * Instead, it uses gridWidth * gridHeight to define the map size.
 */
export interface SOMParams extends BaseClusteringParams {
  /**
   * Width of the SOM grid (number of neurons in x-axis).
   * Must be ≥ 1.
   */
  gridWidth: number;

  /**
   * Height of the SOM grid (number of neurons in y-axis).
   * Must be ≥ 1.
   */
  gridHeight: number;

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
  numEpochs?: number;

  /**
   * Initial learning rate or custom decay function.
   * If number: uses exponential decay from this initial value.
   * If function: custom decay schedule (epoch, totalEpochs) => rate.
   * Default: 0.5
   */
  learningRate?: number | DecayFunction;

  /**
   * Initial neighborhood radius or custom decay function.
   * If number: uses exponential decay from this initial value.
   * If function: custom decay schedule (epoch, totalEpochs) => radius.
   * Default: max(gridWidth, gridHeight) / 2
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
   * When true, allows using partialFit() for streaming data.
   * Default: false
   */
  onlineMode?: boolean;

  /**
   * Mini-batch size for online learning.
   * Only used when onlineMode is true.
   * Default: 32
   */
  miniBatchSize?: number;

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
  totalSamples: number;

  /**
   * Current epoch number (for online learning).
   */
  currentEpoch: number;

  /**
   * Grid dimensions.
   */
  gridWidth: number;
  gridHeight: number;

  /**
   * Configuration parameters.
   */
  params: SOMParams;
}

/**
 * Metrics for evaluating SOM quality.
 */
export interface SOMMetrics {
  /**
   * Average distance between samples and their BMUs.
   */
  quantizationError: number;

  /**
   * Proportion of samples whose BMU and second BMU are not neighbors.
   */
  topographicError: number;
}
