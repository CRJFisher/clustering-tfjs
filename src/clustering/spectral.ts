import type {
  DataMatrix,
  SpectralClusteringParams,
  BaseClustering,
} from './types';
import * as tf from '../backend/adapter';
import {
  compute_rbf_affinity,
  compute_knn_affinity,
  compute_sparse_knn_affinity,
  compute_cosine_affinity,
} from '../graph/affinity';
import {
  SparseMatrix,
  sparse_stats,
  sparse_to_dense_tensor,
} from '../graph/sparse';
import { is_tensor } from '../tensor/tensor_guards';
import type { ClusterRepresentations } from './representations';
import { select_medoids } from './medoid_selection';

// Types for intermediate step results
export interface LaplacianResult {
  laplacian: tf.Tensor2D;
  /** D^{1/2} — square root of the degree vector, matching scipy's dd from csgraph_laplacian. */
  degrees?: tf.Tensor1D;
  /** D^{-1/2} — inverse square root of the degree vector, as returned by normalised_laplacian. */
  sqrt_degrees?: tf.Tensor1D;
}

export interface EmbeddingResult {
  embedding: tf.Tensor2D;
  eigenvalues: tf.Tensor1D;
  raw_eigenvectors?: tf.Tensor2D;
  scaling_factors?: tf.Tensor1D;
}

export interface IntermediateSteps {
  affinity: tf.Tensor2D;
  laplacian: LaplacianResult;
  embedding: EmbeddingResult;
  labels: number[];
}

export interface DebugInfo {
  affinity_stats?: {
    shape: number[];
    nnz: number;
    min: number;
    max: number;
    mean: number;
  };
  laplacian_spectrum?: number[];
  embedding_stats?: {
    shape: number[];
    unique_values_per_dim: number[];
    scaling_factors?: number[];
  };
  clustering_metrics?: {
    inertia: number;
    iterations: number;
  };
}

/**
 * Spectral clustering estimator.
 *
 * Clusters by embedding the data into the eigenspace of its graph Laplacian and
 * running k-means on that embedding. The pipeline is:
 *   1. Build the similarity graph (affinity matrix) — `'rbf'`,
 *      `'nearest_neighbors'`, `'cosine'`, `'precomputed'`, or a user callable.
 *   2. Form the normalised Laplacian and take its smallest eigenvectors.
 *   3. Row-normalise the embedding and cluster it with k-means.
 *
 * Precomputed and callable affinities are validated (square, symmetric,
 * non-negative). The estimator is transductive: it exposes no `predict` and no
 * JSON serialization (representatives are available via {@link compute_medoids}).
 * `dispose()` releases cached tensors; repeated `fit` calls clean up
 * automatically.
 */
export class SpectralClustering
  implements
    BaseClustering<SpectralClusteringParams>,
    ClusterRepresentations
{
  /** Hyper-parameters (deep-copied from user input). */
  public readonly params: SpectralClusteringParams;

  /** Lazy-filled cluster labels after calling `fit`. */
  public labels_: number[] | null = null;

  /**
   * Index of the representative sample (medoid) per cluster, populated by
   * {@link compute_medoids}. Position `c` holds cluster `c`'s medoid index, or
   * `-1` if that cluster has no assigned samples.
   */
  public medoid_indices_: Int32Array | null = null;

  /** Cached affinity matrix (shape: n_samples × n_samples). */
  public affinity_matrix_: tf.Tensor2D | null = null;

  /** Cached sparse affinity matrix for nearest-neighbor fits. */
  public sparse_affinity_matrix_: SparseMatrix | null = null;

  /** Debug information (populated when using return_intermediate_steps) */
  private debug_info_: DebugInfo | null = null;

  /** Whether to capture debug information (modular compatibility) */
  private capture_debug_info: boolean = false;

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
    if (this.affinity_matrix_ != null) {
      this.affinity_matrix_.dispose();
      this.affinity_matrix_ = null;
    }

    this.sparse_affinity_matrix_ = null;
    this.labels_ = null;
    this.medoid_indices_ = null;
  }

  // Allowed affinity options when provided as a string
  private static readonly VALID_AFFINITIES = [
    'rbf',
    'nearest_neighbors',
    'cosine',
    'precomputed',
  ] as const;

  /**
   * @param params - Configuration for spectral clustering.
   */
  constructor(params: SpectralClusteringParams) {
    const { capture_debug_info = false, ...clustering_params } = params;

    // Freeze user params to avoid accidental mutation downstream.
    this.params = { ...clustering_params };
    this.capture_debug_info = capture_debug_info;

    SpectralClustering.validate_params(this.params);
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
    if (this.capture_debug_info) {
      this.debug_info_ = {};
    }

    /* ---------------------------- 0) Input -------------------------------- */
    const x_tensor: tf.Tensor2D =
      is_tensor(_X)
        ? (tf.cast(_X as tf.Tensor2D, 'float32') as tf.Tensor2D)
        : tf.tensor2d(_X as number[][], undefined, 'float32');

    const n_samples = x_tensor.shape[0];
    if (this.params.n_clusters > n_samples) {
      x_tensor.dispose();
      throw new Error('n_clusters cannot exceed number of samples.');
    }

    const use_sparse_nearest_neighbors =
      this.params.affinity === 'nearest_neighbors';
    const max_samples = this.params.max_samples ?? 10_000;
    if (!use_sparse_nearest_neighbors && n_samples > max_samples) {
      x_tensor.dispose();
      throw new Error(
        `Input has ${n_samples} samples, which exceeds the maximum of ${max_samples} ` +
        `for spectral clustering. The algorithm requires O(n^2) memory for the affinity matrix. ` +
        `Set max_samples in params to override this limit if you have sufficient memory.`,
      );
    }

    /* ---------------------------- 1) Affinity ----------------------------- */
    let sparse_affinity: SparseMatrix | null = null;

    if (use_sparse_nearest_neighbors) {
      const k = SpectralClustering.default_neighbors(this.params, n_samples);
      sparse_affinity = compute_sparse_knn_affinity(x_tensor, k, true);
      this.sparse_affinity_matrix_ = sparse_affinity;
    } else {
      this.affinity_matrix_ = SpectralClustering.compute_affinity_matrix(
        x_tensor,
        this.params,
      );
    }

    let affinity_sum: number;
    if (sparse_affinity != null) {
      affinity_sum = 0;
      for (const value of sparse_affinity.data) affinity_sum += value;
    } else {
      const sum_tensor = this.affinity_matrix_!.sum();
      affinity_sum = (await sum_tensor.data())[0];
      sum_tensor.dispose();
    }
    if (affinity_sum === 0) {
      throw new Error(
        'Affinity matrix contains only zeros – cannot perform spectral clustering.',
      );
    }

    // Capture affinity statistics if requested
    if (this.capture_debug_info) {
      if (sparse_affinity != null) {
        this.debug_info_!.affinity_stats = sparse_stats(sparse_affinity);
      } else {
        const data = await this.affinity_matrix_!.data();
        const data_array = Array.from(data);
        const nnz = data_array.filter((v: number) => v !== 0).length;
        this.debug_info_!.affinity_stats = {
          shape: this.affinity_matrix_!.shape,
          nnz,
          min: Math.min(...data_array),
          max: Math.max(...data_array),
          mean:
            data_array.reduce((a: number, b: number) => a + b, 0) /
            data_array.length,
        };
      }
    }

    /* ---------------------------- 2) Component Detection ---------------------- */
    // Detect connected components
    const { detect_connected_components, detect_sparse_connected_components } = await import(
      '../graph/connected_components'
    );
    const { num_components, is_fully_connected, component_labels } =
      sparse_affinity != null
        ? detect_sparse_connected_components(sparse_affinity)
        : detect_connected_components(this.affinity_matrix_ as tf.Tensor2D);

    // Warn if disconnected
    if (!is_fully_connected) {
      console.warn(
        'Graph is not fully connected, spectral embedding may not work as expected.',
      );
    }

    let U: tf.Tensor2D;

    // If graph is disconnected and has enough components, use component indicators
    if (!is_fully_connected && num_components >= this.params.n_clusters) {
      /* ------------------------ Use Component Indicators -------------------- */
      const { create_component_indicators } = await import(
        '../graph/component_indicators'
      );

      // Use all component indicators, not just n_clusters
      // This allows k-means to properly group components into clusters
      U = create_component_indicators(
        component_labels,
        num_components,
        num_components, // Use all components, not this.params.n_clusters
      );

      // Component indicators are already normalized, no scaling needed

      // Capture debug info for component indicators
      if (this.capture_debug_info) {
        // For disconnected components, we don't have a traditional Laplacian spectrum
        // but we can still provide information about the components
        this.debug_info_!.laplacian_spectrum = Array(num_components).fill(0); // Components have eigenvalue 0

        const emb_data = await U.data();
        const [n, k] = U.shape;
        const unique_values_per_dim: number[] = [];

        for (let i = 0; i < k; i++) {
          const col = emb_data.slice(i * n, (i + 1) * n);
          const unique = new Set(col.map((v) => Math.round(v * 1e10) / 1e10));
          unique_values_per_dim.push(unique.size);
        }

        this.debug_info_!.embedding_stats = {
          shape: U.shape,
          unique_values_per_dim,
          scaling_factors: Array(num_components).fill(1), // No scaling for component indicators
        };
      }
    } else {
      /* ---------------------------- Standard Approach ------------------------ */
      const { smallest_eigenvectors_with_values } = await import(
        '../eigen/smallest_eigenvectors_with_values'
      );

      const num_eigenvectors = Math.max(this.params.n_clusters, num_components);
      let U_full: tf.Tensor2D;
      let eigenvalues: tf.Tensor1D;
      let sqrt_degrees_tensor: tf.Tensor1D | null = null;

      if (sparse_affinity != null) {
        const { sparse_normalised_laplacian_operator } = await import(
          '../graph/laplacian'
        );
        const sparse_laplacian =
          sparse_normalised_laplacian_operator(sparse_affinity);

        if (this.capture_debug_info) {
          const spectrum_k = Math.min(10, sparse_laplacian.operator.n);
          const { eigenvalues: spec_evals, eigenvectors: spec_vecs } =
            smallest_eigenvectors_with_values(
              sparse_laplacian.operator,
              spectrum_k,
            );
          const spec_data = await spec_evals.data();
          this.debug_info_!.laplacian_spectrum = Array.from(spec_data);
          spec_evals.dispose();
          spec_vecs.dispose();
        }

        const result = smallest_eigenvectors_with_values(
          sparse_laplacian.operator,
          num_eigenvectors,
        );
        U_full = result.eigenvectors;
        eigenvalues = result.eigenvalues;
        sqrt_degrees_tensor = tf.tensor1d(
          Array.from(sparse_laplacian.sqrt_degrees),
          'float32',
        );
      } else {
        const { normalised_laplacian } = await import('../graph/laplacian');
        const { laplacian, sqrt_degrees } = tf.tidy(() =>
          normalised_laplacian(this.affinity_matrix_ as tf.Tensor2D, true),
        );

        if (this.capture_debug_info) {
          const spectrum_k = Math.min(10, laplacian.shape[0]);
          const { eigenvalues: spec_evals, eigenvectors: spec_vecs } =
            smallest_eigenvectors_with_values(laplacian, spectrum_k);
          const spec_data = await spec_evals.data();
          this.debug_info_!.laplacian_spectrum = Array.from(spec_data);
          spec_evals.dispose();
          spec_vecs.dispose();
        }

        const result = smallest_eigenvectors_with_values(
          laplacian,
          num_eigenvectors,
        );
        U_full = result.eigenvectors;
        eigenvalues = result.eigenvalues;
        sqrt_degrees_tensor = sqrt_degrees;
        laplacian.dispose();
      }

      U = tf.tidy(() => {
        const U_selected = tf.slice(
          U_full,
          [0, 0],
          [-1, this.params.n_clusters],
        ) as tf.Tensor2D;
        const sqrt_deg = tf.pow(sqrt_degrees_tensor!, -1) as tf.Tensor1D;
        const sqrt_deg_col = sqrt_deg.reshape([-1, 1]) as tf.Tensor2D;
        return U_selected.div(sqrt_deg_col) as tf.Tensor2D;
      });

      if (this.capture_debug_info) {
        const emb_data = await U.data();
        const [n, k] = U.shape;
        const unique_values_per_dim: number[] = [];

        for (let i = 0; i < k; i++) {
          const col = emb_data.slice(i * n, (i + 1) * n);
          const unique = new Set(col.map((v) => Math.round(v * 1e10) / 1e10));
          unique_values_per_dim.push(unique.size);
        }

        const eigen_data = await eigenvalues.data();
        this.debug_info_!.embedding_stats = {
          shape: U.shape,
          unique_values_per_dim,
          scaling_factors: Array.from(eigen_data.slice(0, this.params.n_clusters)),
        };
      }

      sqrt_degrees_tensor.dispose();
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
    if (this.params.intensive_parameter_sweep && this.params.affinity === 'rbf') {
      // Intensive parameter sweep for difficult cases
      const { intensive_parameter_sweep } = await import(
        './spectral_optimization'
      );

      let result;
      try {
        result = await intensive_parameter_sweep(
          x_tensor,
          this.params,
          this.compute_embedding_from_affinity.bind(this),
          SpectralClustering.compute_affinity_matrix,
        );
      } catch (err) {
        U.dispose();
        x_tensor.dispose();
        throw err;
      }

      this.labels_ = result.labels;

      // Store debug info
      Object.defineProperty(this, '_debug_intensive_sweep_config_', {
        value: result.config,
        writable: true,
        configurable: true,
        enumerable: false,
      });
    }
    // Check if we should use validation-based optimization
    else if (this.params.use_validation && this.params.n_clusters >= 3) {
      // Use validation metrics to find best clustering
      const { validation_based_optimization } = await import(
        './spectral_optimization'
      );

      const metric = this.params.validation_metric ?? 'calinski-harabasz';
      const attempts = this.params.validation_attempts ?? 20;

      const result = await validation_based_optimization(
        U,
        this.params.n_clusters,
        metric,
        attempts,
        this.params.random_state,
      );

      this.labels_ = result.labels;

      // Store debug info
      Object.defineProperty(this, '_debug_validation_score_', {
        value: result.score,
        writable: true,
        configurable: true,
        enumerable: false,
      });
    } else {
      // Standard k-means without validation
      const km_params = {
        n_clusters: this.params.n_clusters,
        random_state: this.params.random_state,
        // Multiple initialisations significantly increase robustness of the
        // final clustering outcome.  Follow scikit-learn default (n_init = 10)
        // unless the caller supplied an explicit override.
        n_init: this.params.n_init ?? 10,
      } as const;

      const km = new KMeans(km_params);

      // Expose for unit-testing (non-enumerable to avoid polluting logs)
      Object.defineProperty(this, '_debug_last_kmeans_params_', {
        value: km_params,
        writable: true,
        configurable: true,
        enumerable: false,
      });

      // Pass the embedding directly to k-means without row normalization
      await km.fit(U);

      this.labels_ = km.labels_!;

      // Capture clustering metrics if requested
      if (this.capture_debug_info && km.inertia_ !== null) {
        this.debug_info_!.clustering_metrics = {
          inertia: km.inertia_,
          iterations: 0, // KMeans doesn't expose iteration count currently
        };
      }
      km.dispose();
    }

    /* --------------------------- Clean-up --------------------------------- */
    U.dispose();

    x_tensor.dispose();
  }

  /**
   * Fits the model and returns cluster labels.
   *
   * @param X - Input data matrix of shape [n_samples, n_features].
   * @returns Array of cluster labels for each sample.
   * @throws {Error} If n_clusters exceeds n_samples or n_samples exceeds max_samples.
   */
  async fit_predict(X: DataMatrix): Promise<number[]> {
    await this.fit(X);
    if (this.labels_ == null) {
      throw new Error('SpectralClustering failed to compute labels.');
    }
    return this.labels_;
  }

  /**
   * Computes the representative sample (medoid) of every cluster — the sample
   * closest to its cluster mean in the original feature space (Euclidean) — and
   * stores them in {@link medoid_indices_}. SpectralClustering has no synthetic
   * centroids, so medoids are its `ClusterRepresentations` surface.
   *
   * @param X The data the model was fitted on (same row order as `labels_`).
   * @returns The populated `medoid_indices_`.
   * @throws {Error} If called before `fit()`.
   */
  async compute_medoids(X: DataMatrix): Promise<Int32Array> {
    if (this.labels_ == null) {
      throw new Error('SpectralClustering.compute_medoids called before fit().');
    }
    const { indices } = await select_medoids(
      X,
      this.labels_,
      this.params.n_clusters,
      'euclidean',
    );
    this.medoid_indices_ = indices;
    return indices;
  }

  /**
   * Get debug information if available.
   */
  public get_debug_info(): DebugInfo | null {
    return this.debug_info_;
  }

  /**
   * Fits the model and returns intermediate steps for debugging and analysis.
   * This method is useful for comparing with reference implementations.
   */
  async fit_with_intermediate_steps(X: DataMatrix): Promise<IntermediateSteps> {
    // Dispose previous state if the estimator is re-used.
    this.dispose();
    this.debug_info_ = {};

    /* ---------------------------- 0) Input -------------------------------- */
    const x_tensor: tf.Tensor2D =
      is_tensor(X)
        ? (tf.cast(X as tf.Tensor2D, 'float32') as tf.Tensor2D)
        : tf.tensor2d(X as number[][], undefined, 'float32');

    const n_samples_debug = x_tensor.shape[0];
    if (this.params.n_clusters > n_samples_debug) {
      x_tensor.dispose();
      throw new Error('n_clusters cannot exceed number of samples.');
    }
    const use_sparse_nearest_neighbors = this.params.affinity === 'nearest_neighbors';
    const max_samples_debug = this.params.max_samples ?? 10_000;
    if (!use_sparse_nearest_neighbors && n_samples_debug > max_samples_debug) {
      x_tensor.dispose();
      throw new Error(
        `Input has ${n_samples_debug} samples, which exceeds the maximum of ${max_samples_debug} ` +
        `for spectral clustering. The algorithm requires O(n^2) memory for the affinity matrix. ` +
        `Set max_samples in params to override this limit if you have sufficient memory.`,
      );
    }

    /* ---------------------------- 1) Affinity ----------------------------- */
    let sparse_affinity: SparseMatrix | null = null;
    let affinity: tf.Tensor2D;

    if (use_sparse_nearest_neighbors) {
      const k = SpectralClustering.default_neighbors(this.params, n_samples_debug);
      sparse_affinity = compute_sparse_knn_affinity(x_tensor, k, true);
      this.sparse_affinity_matrix_ = sparse_affinity;
      affinity = sparse_to_dense_tensor(sparse_affinity);
    } else {
      affinity = SpectralClustering.compute_affinity_matrix(x_tensor, this.params);
    }

    let affinity_sum: number;
    if (sparse_affinity != null) {
      affinity_sum = 0;
      for (const value of sparse_affinity.data) affinity_sum += value;
    } else {
      const affinity_sum_tensor = affinity.sum();
      affinity_sum = (await affinity_sum_tensor.data())[0];
      affinity_sum_tensor.dispose();
    }
    if (affinity_sum === 0) {
      affinity.dispose();
      x_tensor.dispose();
      throw new Error(
        'Affinity matrix contains only zeros – cannot perform spectral clustering.',
      );
    }

    // Capture affinity statistics
    if (sparse_affinity != null) {
      this.debug_info_.affinity_stats = sparse_stats(sparse_affinity);
    } else {
      const affinity_data = await affinity.data();
      const affinity_array = Array.from(affinity_data);
      const nnz = affinity_array.filter((v: number) => v !== 0).length;
      this.debug_info_.affinity_stats = {
        shape: affinity.shape,
        nnz,
        min: Math.min(...affinity_array),
        max: Math.max(...affinity_array),
        mean:
          affinity_array.reduce((a: number, b: number) => a + b, 0) /
          affinity_array.length,
      };
    }

    /* ---------------------------- 2) Laplacian ----------------------------- */
    const { normalised_laplacian } = await import('../graph/laplacian');
    const { laplacian, sqrt_degrees } = tf.tidy(() =>
      normalised_laplacian(affinity, true),
    );

    // Capture Laplacian spectrum using same solver routing as the main pipeline
    const { smallest_eigenvectors_with_values: spectrum_helper } = await import(
      '../eigen/smallest_eigenvectors_with_values'
    );
    const spectrum_k = Math.min(10, laplacian.shape[0]);
    const { eigenvalues: spec_evals, eigenvectors: spec_vecs } = spectrum_helper(laplacian, spectrum_k);
    const spec_data = await spec_evals.data();
    this.debug_info_.laplacian_spectrum = Array.from(spec_data);
    spec_evals.dispose();
    spec_vecs.dispose();

    /* ---------------------------- 3) Embedding ----------------------------- */
    const { smallest_eigenvectors_with_values } = await import(
      '../eigen/smallest_eigenvectors_with_values'
    );

    const { eigenvectors: U_full, eigenvalues } =
      smallest_eigenvectors_with_values(laplacian, this.params.n_clusters);

    // Apply sklearn's normalization: divide by D^{1/2}
    const embedding = tf.tidy(() => {
      const U_selected = tf.slice(
        U_full,
        [0, 0],
        [-1, this.params.n_clusters],
      ) as tf.Tensor2D;
      // sqrt_degrees is D^{-1/2}, so D^{1/2} = pow(sqrt_degrees, -1)
      const sqrt_deg = tf.pow(sqrt_degrees, -1) as tf.Tensor1D;
      const sqrt_deg_col = sqrt_deg.reshape([-1, 1]) as tf.Tensor2D;
      return U_selected.div(sqrt_deg_col) as tf.Tensor2D;
    });

    // Capture embedding statistics
    const emb_data = await embedding.data();
    const [n, k] = embedding.shape;
    const unique_values_per_dim: number[] = [];

    for (let i = 0; i < k; i++) {
      const col = emb_data.slice(i * n, (i + 1) * n);
      const unique = new Set(col.map((v) => Math.round(v * 1e10) / 1e10));
      unique_values_per_dim.push(unique.size);
    }

    const eigen_data = await eigenvalues.data();
    this.debug_info_.embedding_stats = {
      shape: embedding.shape,
      unique_values_per_dim,
      scaling_factors: Array.from(eigen_data.slice(0, this.params.n_clusters)),
    };

    /* ---------------------------- 4) Clustering ----------------------------- */
    const { KMeans } = await import('./kmeans');
    const km_params = {
      n_clusters: this.params.n_clusters,
      random_state: this.params.random_state,
      n_init: this.params.n_init ?? 10,
    } as const;

    const km = new KMeans(km_params);
    await km.fit(embedding);
    const labels = km.labels_!;

    // Capture clustering metrics
    if (km.inertia_ !== null) {
      this.debug_info_.clustering_metrics = {
        inertia: km.inertia_,
        iterations: 0, // KMeans doesn't expose iteration count currently
      };
    }
    km.dispose();

    /* ---------------------------- Prepare Result ----------------------------- */
    // Compute D^{1/2} for the result (sqrt_degrees is D^{-1/2}, so pow(-1) gives D^{1/2})
    const degrees_intermediate = tf.pow(sqrt_degrees, -1) as tf.Tensor1D;
    const result: IntermediateSteps = {
      affinity: tf.clone(affinity),
      laplacian: {
        laplacian: tf.clone(laplacian),
        degrees: tf.clone(degrees_intermediate),
        sqrt_degrees: tf.clone(sqrt_degrees),
      },
      embedding: {
        embedding: tf.clone(embedding),
        eigenvalues: tf.clone(eigenvalues),
        raw_eigenvectors: tf.clone(U_full),
      },
      labels: [...labels],
    };
    degrees_intermediate.dispose();

    // Store labels for consistency
    this.labels_ = labels;
    if (sparse_affinity == null) {
      this.affinity_matrix_ = tf.clone(affinity);
    }

    /* --------------------------- Clean-up --------------------------------- */
    affinity.dispose();
    laplacian.dispose();
    sqrt_degrees.dispose();
    U_full.dispose();
    eigenvalues.dispose();
    embedding.dispose();

    x_tensor.dispose();

    return result;
  }

  /* ------------------------------------------------------------------- */
  /*                     Static parameter validation                       */
  /* ------------------------------------------------------------------- */

  private static validate_params(params: SpectralClusteringParams): void {
    const { n_clusters, affinity = 'rbf', gamma, n_neighbors } = params;

    // n_clusters must be a positive integer
    if (!Number.isInteger(n_clusters) || n_clusters < 1) {
      throw new Error('n_clusters must be a positive integer (>= 1).');
    }

    // Affinity string or callable
    const is_callable = typeof affinity === 'function';
    if (
      !is_callable &&
      !SpectralClustering.VALID_AFFINITIES.includes(affinity)
    ) {
      throw new Error(
        `Invalid affinity '${affinity}'. Must be one of ${SpectralClustering.VALID_AFFINITIES.join(', ')} or a callable.`,
      );
    }

    // gamma checks (only relevant for RBF affinity when provided as string)
    if (!is_callable && affinity === 'rbf') {
      if (gamma !== undefined && (typeof gamma !== 'number' || gamma <= 0)) {
        throw new Error('gamma must be a positive number if specified.');
      }
    } else if (gamma !== undefined) {
      // If affinity is not RBF but user supplied gamma, warn
      throw new Error("gamma is only applicable when affinity is 'rbf'.");
    }

    // n_neighbors checks for nearest_neighbors affinity
    if (!is_callable && affinity === 'nearest_neighbors') {
      // If n_neighbors is provided, validate it
      if (
        n_neighbors !== undefined &&
        (!Number.isInteger(n_neighbors) || n_neighbors < 1)
      ) {
        throw new Error('n_neighbors must be a positive integer (>= 1).');
      }
      // Default will be computed at fit time based on n_samples
    } else if (n_neighbors !== undefined) {
      throw new Error(
        "n_neighbors is only applicable when affinity is 'nearest_neighbors'.",
      );
    }

    // precomputed: gamma / n_neighbors not allowed
    if (!is_callable && affinity === 'precomputed') {
      if (gamma !== undefined) {
        throw new Error(
          "gamma is not applicable when affinity is 'precomputed'.",
        );
      }
      if (n_neighbors !== undefined) {
        throw new Error(
          "n_neighbors is not applicable when affinity is 'precomputed'.",
        );
      }
    }
  }

  /* ------------------------------------------------------------------- */
  /*                       Affinity matrix utilities                       */
  /* ------------------------------------------------------------------- */

  static compute_affinity_matrix(
    X: tf.Tensor2D,
    params: SpectralClusteringParams,
  ): tf.Tensor2D {
    const { affinity = 'rbf' } = params;

    // -------------------------- Callable affinity ------------------------ //
    if (typeof affinity === 'function') {
      const A = affinity(X);
      SpectralClustering.validate_affinity_matrix(A);
      return A;
    }

    // ---------------------------- Precomputed ---------------------------- //
    if (affinity === 'precomputed') {
      SpectralClustering.validate_affinity_matrix(X);
      return X;
    }

    if (affinity === 'rbf') {
      return compute_rbf_affinity(X, params.gamma);
    }

    if (affinity === 'cosine') {
      return compute_cosine_affinity(X);
    }

    // nearest_neighbors - include self-loops for connectivity
    const n_samples = X.shape[0];
    const k = SpectralClustering.default_neighbors(params, n_samples);
    return compute_knn_affinity(X, k, true);
  }

  /** Returns defaulted k when undefined */
  static default_neighbors(
    params: SpectralClusteringParams,
    n_samples: number,
  ): number {
    if (params.n_neighbors !== undefined) {
      return params.n_neighbors;
    }

    // Match sklearn's default: round(log2(n_samples))
    // Handle edge case: ensure at least 1 neighbor
    const default_k = Math.round(Math.log2(n_samples));
    return Math.max(1, default_k);
  }

  /**
   * Compute spectral embedding from affinity matrix.
   * Extracted to support parameter sweep.
   */
  private async compute_embedding_from_affinity(
    affinity_matrix: tf.Tensor2D,
  ): Promise<tf.Tensor2D> {
    const { detect_connected_components } = await import(
      '../graph/connected_components'
    );
    const { num_components, is_fully_connected, component_labels } =
      detect_connected_components(affinity_matrix);

    if (!is_fully_connected && num_components >= this.params.n_clusters) {
      const { create_component_indicators } = await import(
        '../graph/component_indicators'
      );
      return create_component_indicators(
        component_labels,
        num_components,
        num_components,
      );
    } else {
      const { normalised_laplacian } = await import('../graph/laplacian');
      const { smallest_eigenvectors_with_values } = await import(
        '../eigen/smallest_eigenvectors_with_values'
      );

      const { laplacian, sqrt_degrees } = tf.tidy(() =>
        normalised_laplacian(affinity_matrix, true),
      );

      const num_eigenvectors = Math.max(this.params.n_clusters, num_components);
      const { eigenvectors: U_full, eigenvalues } =
        smallest_eigenvectors_with_values(laplacian, num_eigenvectors);

      const U_scaled = tf.tidy(() => {
        const num_to_use = this.params.n_clusters;
        const U_selected = tf.slice(
          U_full,
          [0, 0],
          [-1, num_to_use],
        ) as tf.Tensor2D;
        // sqrt_degrees is D^{-1/2}, so D^{1/2} = pow(sqrt_degrees, -1)
        const sqrt_deg = tf.pow(sqrt_degrees, -1) as tf.Tensor1D;
        const sqrt_deg_col = sqrt_deg.reshape([-1, 1]) as tf.Tensor2D;
        const U_normalized = U_selected.div(sqrt_deg_col) as tf.Tensor2D;
        return U_normalized;
      });

      laplacian.dispose();
      sqrt_degrees.dispose();
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
  static validate_affinity_matrix(A: tf.Tensor2D): void {
    if (A.shape.length !== 2 || A.shape[0] !== A.shape[1]) {
      throw new Error('Affinity matrix must be square (n × n).');
    }

    // Check symmetry & non-negativity using small tolerances.
    tf.tidy(() => {
      const tol = 1e-6;
      const diff = A.sub(A.transpose()).abs();
      const max_diff = diff.max().dataSync()[0];
      if (max_diff > tol) {
        throw new Error('Affinity matrix must be symmetric.');
      }

      const min_val = A.min().dataSync()[0];
      if (min_val < -tol) {
        throw new Error('Affinity matrix must be non-negative.');
      }
    });
  }
}
