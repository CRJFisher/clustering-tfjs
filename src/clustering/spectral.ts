import type {
  DataMatrix,
  LabelVector,
  SpectralClusteringParams,
  BaseClustering,
} from './types';
import tf from '../tf-adapter';
import { compute_rbf_affinity, compute_knn_affinity } from '../utils/affinity';

// Types for intermediate step results
export interface LaplacianResult {
  laplacian: tf.Tensor2D;
  degrees?: tf.Tensor1D;
  sqrtDegrees?: tf.Tensor1D;
}

export interface EmbeddingResult {
  embedding: tf.Tensor2D;
  eigenvalues: tf.Tensor1D;
  rawEigenvectors?: tf.Tensor2D;
  scalingFactors?: tf.Tensor1D;
}

export interface IntermediateSteps {
  affinity: tf.Tensor2D;
  laplacian: LaplacianResult;
  embedding: EmbeddingResult;
  labels: number[];
}

export interface DebugInfo {
  affinityStats?: {
    shape: number[];
    nnz: number;
    min: number;
    max: number;
    mean: number;
  };
  laplacianSpectrum?: number[];
  embeddingStats?: {
    shape: number[];
    uniqueValuesPerDim: number[];
    scalingFactors?: number[];
  };
  clusteringMetrics?: {
    inertia: number;
    iterations: number;
  };
}

/**
 * Spectral clustering estimator skeleton.
 *
 * This initial implementation only covers:
 *   • Constructor & hyper-parameter validation
 *   • Public instance properties
 *   • Synchronous method stubs for `fit` / `fitPredict`
 *
 * The heavy lifting – affinity matrix construction, graph Laplacian
 * computation, eigen-decomposition and the final k-means step – will be
 * implemented in subsequent tasks (see backlog).
 *
 * Updates introduced in *task-12*:
 *   • Support for `affinity = "precomputed"` and user-supplied callable
 *     affinities with rigorous matrix validation (square, symmetric,
 *     non-negative).
 *   • Public `dispose()` method and automatic clean-up on repeated `fit`
 *     calls to prevent tensor memory leaks.
 */
export class SpectralClustering
  implements BaseClustering<SpectralClusteringParams>
{
  /** Hyper-parameters (deep-copied from user input). */
  public readonly params: SpectralClusteringParams;

  /** Lazy-filled cluster labels after calling `fit`. */
  public labels_: number[] | null = null;

  /** Cached affinity matrix (shape: nSamples × nSamples). */
  public affinityMatrix_: tf.Tensor2D | null = null;

  /** Debug information (populated when using returnIntermediateSteps) */
  private debugInfo_: DebugInfo | null = null;

  /** Whether to capture debug information (modular compatibility) */
  private captureDebugInfo: boolean = false;

  /* ------------------------------------------------------------------- */
  /*                      Resource / memory management                    */
  /* ------------------------------------------------------------------- */

  /**
   * Disposes any tensors kept as instance state and resets internal caches.
   *
   * The estimator instance can still be reused after calling `dispose()` by
   * invoking `fit` again.
   */
  public dispose(): void {
    if (this.affinityMatrix_ != null) {
      this.affinityMatrix_.dispose();
      this.affinityMatrix_ = null;
    }

    if (
      this.labels_ != null &&
      (this.labels_ as unknown as tf.Tensor).dispose instanceof Function
    ) {
      // Only dispose if the labels are a Tensor (not plain array)
      (this.labels_ as unknown as tf.Tensor).dispose();
    }
    this.labels_ = null;
  }

  // Allowed affinity options when provided as a string
  private static readonly VALID_AFFINITIES = [
    'rbf',
    'nearest_neighbors',
    'precomputed',
  ] as const;

  constructor(
    params: SpectralClusteringParams & { captureDebugInfo?: boolean },
  ) {
    // Extract captureDebugInfo if provided (for modular compatibility)
    const { captureDebugInfo = false, ...clusteringParams } = params;

    // Freeze user params to avoid accidental mutation downstream.
    this.params = { ...clusteringParams };
    this.captureDebugInfo = captureDebugInfo;

    SpectralClustering.validateParams(this.params);
  }

  /**
   * Fits the Spectral Clustering model to the input data and stores the
   * resulting cluster labels in {@link labels_}.
   *
   * Pipeline (following scikit-learn implementation):
   *   1. Build similarity graph – affinity matrix A
   *   2. Compute normalised Laplacian L = I − D^{-1/2} A D^{-1/2}
   *   3. Obtain k smallest eigenvectors of L → embedding U (n × k)
   *   4. Run K-Means directly on the rows of U (no row normalization)
   *
   * Note: Row normalization to unit length is only applied when using
   * assign_labels='discretize', not for the default k-means approach.
   */
  async fit(_X: DataMatrix): Promise<void> {
    // Dispose previous state if the estimator is re-used.
    this.dispose();

    // Reset debug info if capturing
    if (this.captureDebugInfo) {
      this.debugInfo_ = {};
    }

    /* ---------------------------- 0) Input -------------------------------- */
    const Xtensor: tf.Tensor2D =
      _X instanceof tf.Tensor
        ? (tf.cast(_X as tf.Tensor2D, 'float32') as tf.Tensor2D)
        : tf.tensor2d(_X as number[][], undefined, 'float32');

    /* ---------------------------- 1) Affinity ----------------------------- */
    this.affinityMatrix_ = SpectralClustering.computeAffinityMatrix(
      Xtensor,
      this.params,
    );

    const affinitySum = (await this.affinityMatrix_.sum().data())[0];
    if (affinitySum === 0) {
      throw new Error(
        'Affinity matrix contains only zeros – cannot perform spectral clustering.',
      );
    }

    // Capture affinity statistics if requested
    if (this.captureDebugInfo) {
      const data = await this.affinityMatrix_.data();
      const dataArray = Array.from(data);
      const nnz = dataArray.filter((v: number) => v !== 0).length;
      this.debugInfo_!.affinityStats = {
        shape: this.affinityMatrix_.shape,
        nnz,
        min: Math.min(...dataArray),
        max: Math.max(...dataArray),
        mean:
          dataArray.reduce((a: number, b: number) => a + b, 0) /
          dataArray.length,
      };
    }

    /* ---------------------------- 2) Component Detection ---------------------- */
    // Detect connected components
    const { detectConnectedComponents } = await import(
      '../utils/connected_components'
    );
    const { numComponents, isFullyConnected, componentLabels } =
      detectConnectedComponents(this.affinityMatrix_ as tf.Tensor2D);

    // Warn if disconnected
    if (!isFullyConnected) {
      console.warn(
        'Graph is not fully connected, spectral embedding may not work as expected.',
      );
    }

    let U: tf.Tensor2D;

    // If graph is disconnected and has enough components, use component indicators
    if (!isFullyConnected && numComponents >= this.params.nClusters) {
      /* ------------------------ Use Component Indicators -------------------- */
      const { createComponentIndicators } = await import(
        '../utils/component_indicators'
      );

      // Use all component indicators, not just nClusters
      // This allows k-means to properly group components into clusters
      U = createComponentIndicators(
        componentLabels,
        numComponents,
        numComponents, // Use all components, not this.params.nClusters
      );

      // Component indicators are already normalized, no scaling needed

      // Capture debug info for component indicators
      if (this.captureDebugInfo) {
        // For disconnected components, we don't have a traditional Laplacian spectrum
        // but we can still provide information about the components
        this.debugInfo_!.laplacianSpectrum = Array(numComponents).fill(0); // Components have eigenvalue 0

        const embData = await U.data();
        const [n, k] = U.shape;
        const uniqueValuesPerDim: number[] = [];

        for (let i = 0; i < k; i++) {
          const col = embData.slice(i * n, (i + 1) * n);
          const unique = new Set(col.map((v) => Math.round(v * 1e10) / 1e10));
          uniqueValuesPerDim.push(unique.size);
        }

        this.debugInfo_!.embeddingStats = {
          shape: U.shape,
          uniqueValuesPerDim,
          scalingFactors: Array(numComponents).fill(1), // No scaling for component indicators
        };
      }
    } else {
      /* ---------------------------- Standard Approach ------------------------ */
      // Compute Laplacian and eigenvectors as before
      const { normalised_laplacian } = await import('../utils/laplacian');

      // Compute normalized Laplacian AND get degree information for recovery
      const { laplacian, sqrtDegrees } = tf.tidy(() =>
        normalised_laplacian(this.affinityMatrix_ as tf.Tensor2D, true),
      );

      // Capture Laplacian spectrum if requested
      if (this.captureDebugInfo) {
        const { jacobi_eigen_decomposition } = await import(
          '../utils/laplacian'
        );
        const { eigenvalues } = await jacobi_eigen_decomposition(laplacian);
        // Take first 10 eigenvalues for spectrum
        const spectrum = eigenvalues.slice(0, Math.min(10, eigenvalues.length));
        this.debugInfo_!.laplacianSpectrum = spectrum;
      }

      // Get eigenvectors AND eigenvalues for diffusion map scaling
      const { smallest_eigenvectors_with_values } = await import(
        '../utils/smallest_eigenvectors_with_values'
      );

      // When we have more components than clusters, we need to get more eigenvectors
      // to ensure we capture all component indicators
      const numEigenvectors = Math.max(this.params.nClusters, numComponents);

      const { eigenvectors: U_full, eigenvalues } =
        smallest_eigenvectors_with_values(laplacian, numEigenvectors);

      // Apply sklearn's normalization: just divide by degree (dd)
      // NO diffusion map scaling for spectral clustering!
      const U_scaled = tf.tidy(() => {
        // For spectral clustering, sklearn uses drop_first=False,
        // so we keep all eigenvectors including the first one

        // However, we only use nClusters eigenvectors for the final clustering
        // If numComponents > nClusters, we still only use nClusters eigenvectors
        // This allows k-means to group multiple components into fewer clusters
        const numToUse = this.params.nClusters;

        // Select the eigenvectors we need
        const U_selected = tf.slice(
          U_full,
          [0, 0],
          [-1, numToUse],
        ) as tf.Tensor2D;

        // Get the degree vector (not sqrt!)
        // sqrtDegrees is D^{-1/2}, so we need to compute D = 1 / (sqrtDegrees^2)
        const degrees = tf.pow(sqrtDegrees, -2) as tf.Tensor1D;

        // sklearn divides by dd (the degree vector)
        // This recovers the embedding from the normalized Laplacian eigenvectors
        const degreesCol = degrees.reshape([-1, 1]) as tf.Tensor2D;
        const U_normalized = U_selected.div(degreesCol) as tf.Tensor2D;

        return U_normalized;
      });

      // Use the scaled eigenvectors
      U = U_scaled;

      // Capture embedding statistics if requested
      if (this.captureDebugInfo) {
        const embData = await U.data();
        const [n, k] = U.shape;
        const uniqueValuesPerDim: number[] = [];

        for (let i = 0; i < k; i++) {
          const col = embData.slice(i * n, (i + 1) * n);
          const unique = new Set(col.map((v) => Math.round(v * 1e10) / 1e10));
          uniqueValuesPerDim.push(unique.size);
        }

        const eigenData = await eigenvalues.data();
        this.debugInfo_!.embeddingStats = {
          shape: U.shape,
          uniqueValuesPerDim,
          scalingFactors: Array.from(eigenData.slice(0, this.params.nClusters)),
        };
      }

      // Clean up intermediate tensors
      laplacian.dispose();
      sqrtDegrees.dispose();
      eigenvalues.dispose();
      U_full.dispose();
    }

    /* -------------------------- 4) K-Means -------------------------------- */
    // IMPORTANT: sklearn does NOT row-normalize when using k-means!
    // Row normalization is only applied when assign_labels='discretize'
    // We pass the embedding directly to k-means without row normalization,
    // matching sklearn's default behavior
    const { KMeans } = await import('./kmeans');

    // Check if we should use intensive parameter sweep
    if (this.params.intensiveParameterSweep && this.params.affinity === 'rbf') {
      // Intensive parameter sweep for difficult cases
      const { intensiveParameterSweep } = await import(
        './spectral_optimization'
      );

      const result = await intensiveParameterSweep(
        Xtensor,
        this.params,
        this.computeEmbeddingFromAffinity.bind(this),
        SpectralClustering.computeAffinityMatrix,
      );

      this.labels_ = result.labels;

      // Store debug info
      Object.defineProperty(this, '_debug_intensive_sweep_config_', {
        value: result.config,
        writable: false,
        configurable: false,
        enumerable: false,
      });
    }
    // Check if we should use validation-based optimization
    else if (this.params.useValidation && this.params.nClusters >= 3) {
      // Use validation metrics to find best clustering
      const { validationBasedOptimization } = await import(
        './spectral_optimization'
      );

      const metric = this.params.validationMetric ?? 'calinski-harabasz';
      const attempts = this.params.validationAttempts ?? 20;

      const result = await validationBasedOptimization(
        U,
        this.params.nClusters,
        metric,
        attempts,
        this.params.randomState,
      );

      this.labels_ = result.labels;

      // Store debug info
      Object.defineProperty(this, '_debug_validation_score_', {
        value: result.score,
        writable: false,
        configurable: false,
        enumerable: false,
      });
    } else {
      // Standard k-means without validation
      const kmParams = {
        nClusters: this.params.nClusters,
        randomState: this.params.randomState,
        // Multiple initialisations significantly increase robustness of the
        // final clustering outcome.  Follow scikit-learn default (nInit = 10)
        // unless the caller supplied an explicit override.
        nInit: this.params.nInit ?? 10,
      } as const;

      const km = new KMeans(kmParams);

      // Expose for unit-testing (non-enumerable to avoid polluting logs)
      Object.defineProperty(this, '_debug_last_kmeans_params_', {
        value: kmParams,
        writable: false,
        configurable: false,
        enumerable: false,
      });

      // Pass the embedding directly to k-means without row normalization
      await km.fit(U);

      this.labels_ = km.labels_ as number[];

      // Capture clustering metrics if requested
      if (this.captureDebugInfo && km.inertia_ !== null) {
        this.debugInfo_!.clusteringMetrics = {
          inertia: km.inertia_,
          iterations: 0, // KMeans doesn't expose iteration count currently
        };
      }
    }

    /* --------------------------- Clean-up --------------------------------- */
    U.dispose();

    if (!(_X instanceof tf.Tensor)) {
      Xtensor.dispose();
    }
  }

  async fitPredict(X: DataMatrix): Promise<LabelVector> {
    await this.fit(X);
    if (this.labels_ == null) {
      throw new Error('SpectralClustering failed to compute labels.');
    }
    return this.labels_;
  }

  /**
   * Get debug information if available.
   */
  public getDebugInfo(): DebugInfo | null {
    return this.debugInfo_;
  }

  /**
   * Fits the model and returns intermediate steps for debugging and analysis.
   * This method is useful for comparing with reference implementations.
   */
  async fitWithIntermediateSteps(X: DataMatrix): Promise<IntermediateSteps> {
    // Dispose previous state if the estimator is re-used.
    this.dispose();
    this.debugInfo_ = {};

    /* ---------------------------- 0) Input -------------------------------- */
    const Xtensor: tf.Tensor2D =
      X instanceof tf.Tensor
        ? (tf.cast(X as tf.Tensor2D, 'float32') as tf.Tensor2D)
        : tf.tensor2d(X as number[][], undefined, 'float32');

    /* ---------------------------- 1) Affinity ----------------------------- */
    const affinity = SpectralClustering.computeAffinityMatrix(
      Xtensor,
      this.params,
    );

    const affinitySum = (await affinity.sum().data())[0];
    if (affinitySum === 0) {
      throw new Error(
        'Affinity matrix contains only zeros – cannot perform spectral clustering.',
      );
    }

    // Capture affinity statistics
    const affinityData = await affinity.data();
    const affinityArray = Array.from(affinityData);
    const nnz = affinityArray.filter((v: number) => v !== 0).length;
    this.debugInfo_.affinityStats = {
      shape: affinity.shape,
      nnz,
      min: Math.min(...affinityArray),
      max: Math.max(...affinityArray),
      mean:
        affinityArray.reduce((a: number, b: number) => a + b, 0) /
        affinityArray.length,
    };

    /* ---------------------------- 2) Laplacian ----------------------------- */
    const { normalised_laplacian } = await import('../utils/laplacian');
    const { laplacian, sqrtDegrees } = tf.tidy(() =>
      normalised_laplacian(affinity, true),
    );

    // Capture Laplacian spectrum
    const { jacobi_eigen_decomposition } = await import('../utils/laplacian');
    const { eigenvalues: laplacianEigenvalues } =
      await jacobi_eigen_decomposition(laplacian);
    const spectrum = laplacianEigenvalues.slice(
      0,
      Math.min(10, laplacianEigenvalues.length),
    );
    this.debugInfo_.laplacianSpectrum = spectrum;

    /* ---------------------------- 3) Embedding ----------------------------- */
    const { smallest_eigenvectors_with_values } = await import(
      '../utils/smallest_eigenvectors_with_values'
    );

    const { eigenvectors: U_full, eigenvalues } =
      smallest_eigenvectors_with_values(laplacian, this.params.nClusters);

    // Apply sklearn's normalization
    const embedding = tf.tidy(() => {
      const U_selected = tf.slice(
        U_full,
        [0, 0],
        [-1, this.params.nClusters],
      ) as tf.Tensor2D;
      const degrees = tf.pow(sqrtDegrees, -2) as tf.Tensor1D;
      const degreesCol = degrees.reshape([-1, 1]) as tf.Tensor2D;
      return U_selected.div(degreesCol) as tf.Tensor2D;
    });

    // Capture embedding statistics
    const embData = await embedding.data();
    const [n, k] = embedding.shape;
    const uniqueValuesPerDim: number[] = [];

    for (let i = 0; i < k; i++) {
      const col = embData.slice(i * n, (i + 1) * n);
      const unique = new Set(col.map((v) => Math.round(v * 1e10) / 1e10));
      uniqueValuesPerDim.push(unique.size);
    }

    const eigenData = await eigenvalues.data();
    this.debugInfo_.embeddingStats = {
      shape: embedding.shape,
      uniqueValuesPerDim,
      scalingFactors: Array.from(eigenData.slice(0, this.params.nClusters)),
    };

    /* ---------------------------- 4) Clustering ----------------------------- */
    const { KMeans } = await import('./kmeans');
    const kmParams = {
      nClusters: this.params.nClusters,
      randomState: this.params.randomState,
      nInit: this.params.nInit ?? 10,
    } as const;

    const km = new KMeans(kmParams);
    await km.fit(embedding);
    const labels = km.labels_ as number[];

    // Capture clustering metrics
    if (km.inertia_ !== null) {
      this.debugInfo_.clusteringMetrics = {
        inertia: km.inertia_,
        iterations: 0, // KMeans doesn't expose iteration count currently
      };
    }

    /* ---------------------------- Prepare Result ----------------------------- */
    const result: IntermediateSteps = {
      affinity: tf.clone(affinity),
      laplacian: {
        laplacian: tf.clone(laplacian),
        degrees: tf.clone(tf.pow(sqrtDegrees, -2) as tf.Tensor1D),
        sqrtDegrees: tf.clone(sqrtDegrees),
      },
      embedding: {
        embedding: tf.clone(embedding),
        eigenvalues: tf.clone(eigenvalues),
        rawEigenvectors: tf.clone(U_full),
      },
      labels: [...labels],
    };

    // Store labels for consistency
    this.labels_ = labels;
    this.affinityMatrix_ = tf.clone(affinity);

    /* --------------------------- Clean-up --------------------------------- */
    affinity.dispose();
    laplacian.dispose();
    sqrtDegrees.dispose();
    U_full.dispose();
    eigenvalues.dispose();
    embedding.dispose();

    if (!(X instanceof tf.Tensor)) {
      Xtensor.dispose();
    }

    return result;
  }

  /* ------------------------------------------------------------------- */
  /*                     Static parameter validation                       */
  /* ------------------------------------------------------------------- */

  private static validateParams(params: SpectralClusteringParams): void {
    const { nClusters, affinity = 'rbf', gamma, nNeighbors } = params;

    // nClusters must be a positive integer
    if (!Number.isInteger(nClusters) || nClusters < 1) {
      throw new Error('nClusters must be a positive integer (>= 1).');
    }

    // Affinity string or callable
    const isCallable = typeof affinity === 'function';
    if (
      !isCallable &&
      !SpectralClustering.VALID_AFFINITIES.includes(affinity)
    ) {
      throw new Error(
        `Invalid affinity '${affinity}'. Must be one of ${SpectralClustering.VALID_AFFINITIES.join(', ')} or a callable.`,
      );
    }

    // gamma checks (only relevant for RBF affinity when provided as string)
    if (!isCallable && affinity === 'rbf') {
      if (gamma !== undefined && (typeof gamma !== 'number' || gamma <= 0)) {
        throw new Error('gamma must be a positive number if specified.');
      }
    } else if (gamma !== undefined) {
      // If affinity is not RBF but user supplied gamma, warn
      throw new Error("gamma is only applicable when affinity is 'rbf'.");
    }

    // nNeighbors checks for nearest_neighbors affinity
    if (!isCallable && affinity === 'nearest_neighbors') {
      // If nNeighbors is provided, validate it
      if (
        nNeighbors !== undefined &&
        (!Number.isInteger(nNeighbors) || nNeighbors < 1)
      ) {
        throw new Error('nNeighbors must be a positive integer (>= 1).');
      }
      // Default will be computed at fit time based on n_samples
    } else if (nNeighbors !== undefined) {
      throw new Error(
        "nNeighbors is only applicable when affinity is 'nearest_neighbors'.",
      );
    }

    // precomputed: gamma / nNeighbors not allowed
    if (!isCallable && affinity === 'precomputed') {
      if (gamma !== undefined) {
        throw new Error(
          "gamma is not applicable when affinity is 'precomputed'.",
        );
      }
      if (nNeighbors !== undefined) {
        throw new Error(
          "nNeighbors is not applicable when affinity is 'precomputed'.",
        );
      }
    }
  }

  /* ------------------------------------------------------------------- */
  /*                       Affinity matrix utilities                       */
  /* ------------------------------------------------------------------- */

  static computeAffinityMatrix(
    X: tf.Tensor2D,
    params: SpectralClusteringParams,
  ): tf.Tensor2D {
    const { affinity = 'rbf' } = params;

    // -------------------------- Callable affinity ------------------------ //
    if (typeof affinity === 'function') {
      const A = affinity(X);
      SpectralClustering.validateAffinityMatrix(A);
      return A;
    }

    // ---------------------------- Precomputed ---------------------------- //
    if (affinity === 'precomputed') {
      SpectralClustering.validateAffinityMatrix(X);
      return X;
    }

    if (affinity === 'rbf') {
      return compute_rbf_affinity(X, params.gamma);
    }

    // nearest_neighbors - include self-loops for connectivity
    const nSamples = X.shape[0];
    const k = SpectralClustering.defaultNeighbors(params, nSamples);
    return compute_knn_affinity(X, k, true);
  }

  /** Returns defaulted k when undefined */
  static defaultNeighbors(
    params: SpectralClusteringParams,
    nSamples: number,
  ): number {
    if (params.nNeighbors !== undefined) {
      return params.nNeighbors;
    }

    // Match sklearn's default: round(log2(n_samples))
    // Handle edge case: ensure at least 1 neighbor
    const defaultK = Math.round(Math.log2(nSamples));
    return Math.max(1, defaultK);
  }

  /**
   * Compute spectral embedding from affinity matrix.
   * Extracted to support parameter sweep.
   */
  private async computeEmbeddingFromAffinity(
    affinityMatrix: tf.Tensor2D,
  ): Promise<tf.Tensor2D> {
    const { detectConnectedComponents } = await import(
      '../utils/connected_components'
    );
    const { numComponents, isFullyConnected, componentLabels } =
      detectConnectedComponents(affinityMatrix);

    if (!isFullyConnected && numComponents >= this.params.nClusters) {
      const { createComponentIndicators } = await import(
        '../utils/component_indicators'
      );
      return createComponentIndicators(
        componentLabels,
        numComponents,
        numComponents,
      );
    } else {
      const { normalised_laplacian } = await import('../utils/laplacian');
      const { smallest_eigenvectors_with_values } = await import(
        '../utils/smallest_eigenvectors_with_values'
      );

      const { laplacian, sqrtDegrees } = tf.tidy(() =>
        normalised_laplacian(affinityMatrix, true),
      );

      const numEigenvectors = Math.max(this.params.nClusters, numComponents);
      const { eigenvectors: U_full, eigenvalues } =
        smallest_eigenvectors_with_values(laplacian, numEigenvectors);

      const U_scaled = tf.tidy(() => {
        const numToUse = this.params.nClusters;
        const U_selected = tf.slice(
          U_full,
          [0, 0],
          [-1, numToUse],
        ) as tf.Tensor2D;
        const degrees = tf.pow(sqrtDegrees, -2) as tf.Tensor1D;
        const degreesCol = degrees.reshape([-1, 1]) as tf.Tensor2D;
        const U_normalized = U_selected.div(degreesCol) as tf.Tensor2D;
        return U_normalized;
      });

      laplacian.dispose();
      sqrtDegrees.dispose();
      eigenvalues.dispose();
      U_full.dispose();

      return U_scaled;
    }
  }

  /**
   * Validates that the provided tensor is a proper affinity / similarity
   * matrix suitable for spectral clustering.
   *   • Must be 2-D & **square**
   *   • Must be **symmetric** (within tolerance)
   *   • Must be **non-negative** (entries ≥ 0)
   */
  static validateAffinityMatrix(A: tf.Tensor2D): void {
    if (A.shape.length !== 2 || A.shape[0] !== A.shape[1]) {
      throw new Error('Affinity matrix must be square (n × n).');
    }

    // Check symmetry & non-negativity using small tolerances.
    tf.tidy(() => {
      const tol = 1e-6;
      const diff = A.sub(A.transpose()).abs();
      const maxDiff = diff.max().dataSync()[0];
      if (maxDiff > tol) {
        throw new Error('Affinity matrix must be symmetric.');
      }

      const minVal = A.min().dataSync()[0];
      if (minVal < -tol) {
        throw new Error('Affinity matrix must be non-negative.');
      }
    });
  }
}
