import * as tf from "@tensorflow/tfjs-node";

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
    | "rbf"
    | "nearest_neighbors"
    | "precomputed"
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
   * scikit-learn’s `n_init` parameter. If omitted the algorithm defaults to
   * **10** which yields considerably more robust cluster assignments on
   * challenging spectra than a single initialisation.
   */
  nInit?: number;
}

export interface AgglomerativeClusteringParams extends BaseClusteringParams {
  /**
   * Linkage criterion used to merge clusters.
   */
  linkage?: "ward" | "complete" | "average" | "single";

  /**
   * Metric to compute linkage.
   */
  metric?: "euclidean" | "manhattan" | "cosine";
}

/**
 * ----------------------------------------------------------------------------
 * Base interface to be implemented by *all* clustering estimators
 * ----------------------------------------------------------------------------
 */

export interface BaseClustering<Params extends BaseClusteringParams = BaseClusteringParams> {
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
