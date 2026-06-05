import * as tf from '../backend/adapter';
import type {
  BaseClustering,
  DataMatrix,
  SOMParams,
  SOMState,
  SOMTopology,
  SOMNeighborhood,
  SOMInitialization,
  SOMClusterOptions,
  DecayFunction,
} from './types';
import { AgglomerativeClustering } from './agglomerative';
import { is_tensor } from '../tensor/tensor_guards';
import { make_random_stream, type RandomStream } from '../random';
import {
  initialize_weights,
  find_bmu_batch,
  compute_neighborhood_influence_batch,
  create_grid_distance_matrix,
  compute_bmu_distances,
  create_decay_scheduler,
  validate_neighborhood_params,
} from './som_neighborhood';

/**
 * Self-Organizing Map (SOM) implementation using TensorFlow.js.
 * 
 * SOMs create a low-dimensional (typically 2D) discrete representation
 * of high-dimensional input space while preserving topological properties.
 */
export class SOM implements BaseClustering<SOMParams> {
  public readonly params: SOMParams;
  
  // Model state
  public weights_: tf.Tensor3D | null = null;
  public labels_: number[] | null = null;
  public bmus_: tf.Tensor2D | null = null;
  
  // Training state
  private grid_distance_matrix_: tf.Tensor2D | null = null;
  private learning_rate_scheduler_: DecayFunction | null = null;
  private radius_scheduler_: DecayFunction | null = null;
  private total_samples_learned_: number = 0;
  private last_batch_size_: number = 0;
  private current_epoch_: number = 0;
  private quantization_errors_: number[] = [];
  
  // Default parameters
  private static readonly DEFAULT_TOPOLOGY: SOMTopology = 'rectangular';
  private static readonly DEFAULT_NEIGHBORHOOD: SOMNeighborhood = 'gaussian';
  private static readonly DEFAULT_NUM_EPOCHS = 100;
  private static readonly DEFAULT_LEARNING_RATE = 0.5;
  private static readonly DEFAULT_INITIALIZATION: SOMInitialization = 'random';
  private static readonly DEFAULT_TOL = 1e-4;
  private static readonly DEFAULT_MINI_BATCH_SIZE = 32;
  
  constructor(params: SOMParams) {
    this.params = this.validate_and_complete_params(params);
    this.initialize_schedulers();
  }
  
  /**
   * Validate and set default parameters.
   */
  private validate_and_complete_params(params: SOMParams): SOMParams {
    if (!params.grid_width || params.grid_width < 1) {
      throw new Error('gridWidth must be >= 1');
    }
    if (!params.grid_height || params.grid_height < 1) {
      throw new Error('gridHeight must be >= 1');
    }
    
    return {
      ...params,
      topology: params.topology ?? SOM.DEFAULT_TOPOLOGY,
      neighborhood: params.neighborhood ?? SOM.DEFAULT_NEIGHBORHOOD,
      num_epochs: params.num_epochs ?? SOM.DEFAULT_NUM_EPOCHS,
      learning_rate: params.learning_rate ?? SOM.DEFAULT_LEARNING_RATE,
      initialization: params.initialization ?? SOM.DEFAULT_INITIALIZATION,
      tol: params.tol ?? SOM.DEFAULT_TOL,
      mini_batch_size: params.mini_batch_size ?? SOM.DEFAULT_MINI_BATCH_SIZE,
      online_mode: params.online_mode ?? false,
    };
  }
  
  /**
   * Initialize learning rate and radius schedulers.
   */
  private initialize_schedulers(): void {
    const { grid_width, grid_height, num_epochs, learning_rate, radius } = this.params;
    
    // Learning rate scheduler
    if (typeof learning_rate === 'function') {
      this.learning_rate_scheduler_ = learning_rate;
    } else {
      this.learning_rate_scheduler_ = create_decay_scheduler(
        learning_rate as number,
        'exponential',
        num_epochs!
      );
    }
    
    // Radius scheduler
    if (radius !== undefined) {
      if (typeof radius === 'function') {
        this.radius_scheduler_ = radius;
      } else {
        this.radius_scheduler_ = create_decay_scheduler(
          radius,
          'exponential',
          num_epochs!
        );
      }
    } else {
      // Default: adaptive radius based on grid size
      const initial_radius = Math.max(grid_width, grid_height) / 2;
      this.radius_scheduler_ = create_decay_scheduler(
        initial_radius,
        'exponential',
        num_epochs!,
        1 // final radius
      );
    }
  }
  
  /**
   * Fit the SOM to the provided data.
   */
  async fit(X: DataMatrix): Promise<void> {
    const x_tensor = is_tensor(X) ? X as tf.Tensor2D : tf.tensor2d(X);
    
    try {
      await this.fit_tensor(x_tensor);
    } finally {
      if (!is_tensor(X)) {
        x_tensor.dispose();
      }
    }
  }
  
  /**
   * Internal fit method using tensors.
   */
  private async fit_tensor(X: tf.Tensor2D): Promise<void> {
    const { 
      grid_width, 
      grid_height, 
      topology, 
      num_epochs, 
      initialization, 
      random_state,
      tol
    } = this.params;
    
    const [n_samples, _n_features] = X.shape;
    
    // Initialize weights if not already done
    if (!this.weights_) {
      this.weights_ = initialize_weights(
        X,
        grid_height,
        grid_width,
        initialization!,
        random_state
      );
    }
    
    // Pre-compute grid distance matrix
    if (!this.grid_distance_matrix_) {
      this.grid_distance_matrix_ = create_grid_distance_matrix(
        grid_height,
        grid_width,
        topology!
      );
    }
    
    // Training loop
    let prev_quantization_error = Infinity;
    this.quantization_errors_ = [];
    const rng = make_random_stream(random_state);

    for (let epoch = 0; epoch < num_epochs!; epoch++) {
      this.current_epoch_ = epoch;

      // Get current learning rate and radius
      const current_learning_rate = this.learning_rate_scheduler_!(epoch, num_epochs!);
      const current_radius = this.radius_scheduler_!(epoch, num_epochs!);

      // Validate neighborhood parameters
      validate_neighborhood_params(current_radius, grid_height, grid_width);

      // Shuffle data each epoch to avoid order-dependent bias
      const shuffled_indices = this.shuffle_indices(n_samples, rng);
      const indices_tensor = tf.tensor1d(shuffled_indices, 'int32');
      const shuffled_x = tf.gather(X, indices_tensor) as tf.Tensor2D;
      indices_tensor.dispose();

      // Process batch
      const { quantization_error } = await this.train_epoch(
        shuffled_x,
        current_learning_rate,
        current_radius
      );

      shuffled_x.dispose();

      this.quantization_errors_.push(quantization_error);

      // Check convergence
      if (Math.abs(prev_quantization_error - quantization_error) < tol!) {
        break;
      }

      prev_quantization_error = quantization_error;

      // Update total samples learned (only for online mode)
      if (this.params.online_mode) {
        this.total_samples_learned_ += n_samples;
      }
    }
    
    // Compute final BMUs and labels
    await this.compute_final_labels(X);
  }
  
  /**
   * Train one epoch.
   */
  private async train_epoch(
    X: tf.Tensor2D,
    learning_rate: number,
    radius: number
  ): Promise<{ quantization_error: number }> {
    const { neighborhood, mini_batch_size } = this.params;
    const [n_samples] = X.shape;
    
    // Process in mini-batches for memory efficiency
    const batch_size = Math.min(mini_batch_size!, n_samples);
    let total_quantization_error = 0;
    let samples_processed = 0;
    
    for (let i = 0; i < n_samples; i += batch_size) {
      const end_idx = Math.min(i + batch_size, n_samples);
      const batch_x = X.slice([i, 0], [end_idx - i, -1]);
      
      // Find BMUs for batch
      const bmus = find_bmu_batch(batch_x, this.weights_!);
      
      // Get BMU flat indices
      const bmu_indices = tf.tidy(() => {
        const bmus_data = bmus.arraySync();
        const indices = bmus_data.map(([row, col]) => 
          row * this.params.grid_width + col
        );
        return tf.tensor1d(indices, 'int32');
      });
      
      // Compute neighborhood influence
      const influence = compute_neighborhood_influence_batch(
        bmu_indices,
        this.grid_distance_matrix_!,
        radius,
        neighborhood!
      );
      
      // Update weights
      this.update_weights(batch_x, influence, learning_rate);
      
      // Compute quantization error for this batch
      const distances = compute_bmu_distances(batch_x, this.weights_!, bmus);
      const batch_error = distances.mean().arraySync() as number;
      total_quantization_error += batch_error * (end_idx - i);
      samples_processed += (end_idx - i);
      
      // Clean up
      batch_x.dispose();
      bmus.dispose();
      bmu_indices.dispose();
      influence.dispose();
      distances.dispose();
    }
    
    return {
      quantization_error: total_quantization_error / samples_processed
    };
  }
  
  /**
   * Update weights based on samples and neighborhood influence.
   */
  private update_weights(
    samples: tf.Tensor2D,
    influence: tf.Tensor2D,
    learning_rate: number
  ): void {
    tf.tidy(() => {
      const [_n_samples, n_features] = samples.shape;
      const [grid_height, grid_width, _n_features_weight] = this.weights_!.shape;
      const total_neurons = grid_height * grid_width;
      
      // Reshape weights for update
      const weights_flat = this.weights_!.reshape([total_neurons, n_features]);
      
      // Batch SOM update for each neuron j:
      // Δw_j = lr * Σ_i(h_ij * (x_i - w_j)) / Σ_i(h_ij)
      // Normalizes by sum of influences to make updates independent of batch size

      // Expand samples for broadcasting
      const samples_expanded = samples.expandDims(1); // [nSamples, 1, nFeatures]
      const weights_expanded = weights_flat.expandDims(0); // [1, totalNeurons, nFeatures]

      // Compute differences
      const diff = samples_expanded.sub(weights_expanded); // [nSamples, totalNeurons, nFeatures]

      // Apply influence
      const influence_expanded = influence.expandDims(2); // [nSamples, totalNeurons, 1]
      const weighted_diff = diff.mul(influence_expanded);

      // Sum over samples
      const total_update = weighted_diff.sum(0); // [totalNeurons, nFeatures]

      // Normalize by sum of influences per neuron (sign-preserving for mexican_hat)
      const influence_sum = influence.sum(0); // [totalNeurons]
      const epsilon = 1e-8;
      const abs_influence_sum = influence_sum.abs();
      const influence_sum_safe = tf.where(
        abs_influence_sum.greater(epsilon),
        influence_sum,
        tf.fill(influence_sum.shape, epsilon)
      );
      const normalized_update = total_update.div(influence_sum_safe.expandDims(1));

      // Apply updates with learning rate
      const new_weights_flat = weights_flat.add(normalized_update.mul(learning_rate));
      
      // Reshape back to grid
      const new_weights = new_weights_flat.reshape([grid_height, grid_width, n_features]) as tf.Tensor3D;
      
      // Update weights in place (keep new_weights from being disposed by tidy)
      this.weights_!.dispose();
      this.weights_ = tf.keep(new_weights);
    });
  }
  
  /**
   * Compute final BMUs and labels after training.
   */
  private async compute_final_labels(X: tf.Tensor2D): Promise<void> {
    this.bmus_ = find_bmu_batch(X, this.weights_!);
    
    // Convert BMUs to 1D labels
    const bmus_data = await this.bmus_.array();
    const labels = bmus_data.map(([row, col]) => 
      row * this.params.grid_width + col
    );
    
    this.labels_ = labels;
  }
  
  /**
   * Fit and return predicted labels.
   */
  async fit_predict(X: DataMatrix): Promise<number[]> {
    await this.fit(X);
    return this.labels_!;
  }
  
  /**
   * Predict labels for new data using trained SOM.
   */
  async predict(X: DataMatrix): Promise<number[]> {
    if (!this.weights_) {
      throw new Error('SOM must be fitted before prediction');
    }
    
    const x_tensor = is_tensor(X) ? X as tf.Tensor2D : tf.tensor2d(X);
    
    try {
      const bmus = find_bmu_batch(x_tensor, this.weights_);
      const bmus_data = await bmus.array();
      const labels = bmus_data.map(([row, col]) => 
        row * this.params.grid_width + col
      );
      bmus.dispose();
      return labels;
    } finally {
      if (!is_tensor(X)) {
        x_tensor.dispose();
      }
    }
  }
  
  /**
   * Perform 2-phase clustering to produce meaningful cluster labels.
   *
   * SOM neurons outnumber the desired clusters (grid_width * grid_height >> n_clusters),
   * so raw BMU indices are not useful as cluster assignments. This method applies
   * agglomerative (hierarchical) clustering on the trained SOM weight vectors to
   * group neurons into `n_clusters` macro-clusters, then maps each data point's
   * BMU index to its corresponding macro-cluster label.
   *
   * Phase 1: SOM training (must already be completed via fit/fit_predict).
   * Phase 2: AgglomerativeClustering on the [grid_height * grid_width, n_features]
   * weight matrix, producing n_clusters neuron groups.
   *
   * @param n_clusters - Desired number of output clusters. Must be a positive
   *   integer >= 1 and <= grid_width * grid_height.
   * @param options - Optional agglomerative clustering parameters.
   * @param options.linkage - Linkage criterion for merging neuron clusters.
   *   One of 'ward', 'complete', 'average', or 'single'. Default: 'ward'.
   * @param options.metric - Distance metric for linkage computation.
   *   One of 'euclidean', 'manhattan', or 'cosine'. Default: 'euclidean'.
   * @returns Array of cluster labels (0 to n_clusters-1), one per sample from
   *   the most recent fit/fit_predict call.
   * @throws Error if the SOM has not been fitted yet.
   * @throws Error if n_clusters is not a positive integer or exceeds grid_width * grid_height.
   *
   * @example
   * ```typescript
   * const som = new SOM({
   *   grid_width: 5,
   *   grid_height: 5,
   *   num_epochs: 100,
   *   random_state: 42,
   * });
   *
   * const data = [[0, 0], [1, 1], [5, 5], [6, 6], [10, 10], [11, 11]];
   * await som.fit(data);
   *
   * // Get 3 meaningful clusters from the 25-neuron grid
   * const labels = await som.cluster(3);
   * // labels: [0, 0, 1, 2, 2, ...] — one per data point
   *
   * // With custom linkage
   * const labels2 = await som.cluster(4, { linkage: 'average' });
   * ```
   */
  async cluster(
    n_clusters: number,
    options?: SOMClusterOptions
  ): Promise<number[]> {
    if (!this.weights_ || !this.labels_) {
      throw new Error('SOM must be fitted before clustering. Call fit() first.');
    }

    const { grid_height, grid_width } = this.params;
    const total_neurons = grid_height * grid_width;

    if (!Number.isInteger(n_clusters) || n_clusters < 1) {
      throw new Error('nClusters must be a positive integer (>= 1).');
    }
    if (n_clusters > total_neurons) {
      throw new Error(
        `nClusters (${n_clusters}) exceeds total number of neurons (${total_neurons}). Maximum is gridWidth * gridHeight.`
      );
    }

    // Flatten weight grid [grid_height, grid_width, n_features] -> [total_neurons, n_features]
    const weights_data = this.weights_.arraySync();
    const neuron_vectors: number[][] = [];
    for (let row = 0; row < grid_height; row++) {
      for (let col = 0; col < grid_width; col++) {
        neuron_vectors.push(weights_data[row][col]);
      }
    }

    // Run agglomerative clustering on neuron weight vectors
    const agglo = new AgglomerativeClustering({
      n_clusters,
      linkage: options?.linkage ?? 'ward',
      metric: options?.metric ?? 'euclidean',
    });

    await agglo.fit(neuron_vectors);
    const neuron_labels = agglo.labels_!;

    // Map each data point's BMU flat index to its neuron cluster label
    return this.labels_.map(bmu_index => neuron_labels[bmu_index]);
  }

  /**
   * Partial fit for online/incremental learning.
   * Continues training from current state.
   *
   * On the first call (when no weights exist), the input dimensionality establishes
   * the expected feature count. All subsequent calls must provide data with the
   * same number of features; a mismatch throws an error.
   *
   * @param X - Data matrix of shape [n_samples, n_features]. The n_features dimension
   *   must match the feature count from prior fit() or partial_fit() calls.
   * @throws Error if online_mode is not enabled.
   * @throws Error if n_features does not match the feature dimensionality of existing weights.
   */
  async partial_fit(X: DataMatrix): Promise<void> {
    if (!this.params.online_mode) {
      throw new Error('partialFit requires onlineMode to be enabled');
    }
    
    const x_tensor = is_tensor(X) ? X as tf.Tensor2D : tf.tensor2d(X);
    
    try {
      const [n_samples] = x_tensor.shape;
      this.last_batch_size_ = n_samples;

      // Initialize if first call
      if (!this.weights_) {
        // Initialize weights and grid
        this.weights_ = initialize_weights(
          x_tensor,
          this.params.grid_height,
          this.params.grid_width,
          this.params.initialization!,
          this.params.random_state
        );
        
        this.grid_distance_matrix_ = create_grid_distance_matrix(
          this.params.grid_height,
          this.params.grid_width,
          this.params.topology!
        );
        
        // Initialize schedulers
        this.initialize_schedulers();
      } else {
        // Validate feature dimensions match existing weights
        const expected_features = this.weights_.shape[2];
        const actual_features = x_tensor.shape[1];
        if (actual_features !== expected_features) {
          throw new Error(
            `Feature dimension mismatch: expected ${expected_features} features to match prior fit, but got ${actual_features}`
          );
        }
      }

      // Get current learning rate and radius based on total samples learned
      const virtual_epoch = Math.floor(
        this.total_samples_learned_ / n_samples
      );
      const current_learning_rate = this.learning_rate_scheduler_!(
        virtual_epoch,
        this.params.num_epochs!
      );
      const current_radius = this.radius_scheduler_!(
        virtual_epoch,
        this.params.num_epochs!
      );
      
      // Train on batch
      await this.train_epoch(x_tensor, current_learning_rate, current_radius);
      
      // Update total samples learned
      this.total_samples_learned_ += n_samples;
      
      // Update labels
      await this.compute_final_labels(x_tensor);
    } finally {
      if (!is_tensor(X)) {
        x_tensor.dispose();
      }
    }
  }
  
  /**
   * Returns the trained weight vectors of all neurons as a plain JavaScript array.
   *
   * The returned array has shape `[grid_height][grid_width][n_features]` where each
   * element `weights[row][col]` is the weight vector (codebook entry) for the
   * neuron at grid position `(row, col)`.
   *
   * The returned array is a snapshot (deep copy) of the current internal state.
   * Mutating it will not affect the SOM, and calling {@link dispose} will not
   * invalidate it.
   *
   * @returns Plain 3D array of neuron weight vectors `[grid_height][grid_width][n_features]`.
   * @throws Error if the SOM has not been fitted yet.
   */
  get_weights(): number[][][] {
    if (!this.weights_) {
      throw new Error('SOM must be fitted first');
    }
    return this.weights_.arraySync();
  }
  
  /**
   * Calculate the U-matrix (unified distance matrix).
   * Shows the average distance between each neuron and its neighbors.
   */
  get_u_matrix(): tf.Tensor2D {
    if (!this.weights_) {
      throw new Error('SOM must be fitted first');
    }
    
    return tf.tidy(() => {
      const { grid_height, grid_width, topology } = this.params;
      const u_matrix = tf.buffer([grid_height, grid_width]);
      const weights_data = this.weights_!.arraySync();
      
      for (let i = 0; i < grid_height; i++) {
        for (let j = 0; j < grid_width; j++) {
          const current_weight = weights_data[i][j];
          let total_distance = 0;
          let neighbor_count = 0;
          
          // Get neighbors based on topology
          let neighbors: number[][];
          
          if (topology === 'rectangular') {
            // 8-connected rectangular grid
            neighbors = [
              [i - 1, j], [i + 1, j],
              [i, j - 1], [i, j + 1],
              [i - 1, j - 1], [i - 1, j + 1],
              [i + 1, j - 1], [i + 1, j + 1]
            ];
          } else {
            // Hexagonal grid (6-connected)
            const even_row = i % 2 === 0;
            neighbors = even_row ? [
              [i - 1, j - 1], [i - 1, j],  // Top-left, top-right
              [i, j - 1], [i, j + 1],      // Left, right
              [i + 1, j - 1], [i + 1, j]   // Bottom-left, bottom-right
            ] : [
              [i - 1, j], [i - 1, j + 1],  // Top-left, top-right
              [i, j - 1], [i, j + 1],      // Left, right
              [i + 1, j], [i + 1, j + 1]   // Bottom-left, bottom-right
            ];
          }
          
          for (const [ni, nj] of neighbors) {
            if (ni >= 0 && ni < grid_height && nj >= 0 && nj < grid_width) {
              const neighbor_weight = weights_data[ni][nj];
              const distance = Math.sqrt(
                current_weight.reduce((sum, val, idx) => 
                  sum + Math.pow(val - neighbor_weight[idx], 2), 0
                )
              );
              total_distance += distance;
              neighbor_count++;
            }
          }
          
          u_matrix.set(
            neighbor_count > 0 ? total_distance / neighbor_count : 0,
            i,
            j
          );
        }
      }
      
      return u_matrix.toTensor() as tf.Tensor2D;
    });
  }
  
  /**
   * Calculate quantization error.
   * Average distance between samples and their BMUs.
   */
  quantization_error(): number {
    if (this.quantization_errors_.length === 0) {
      throw new Error('SOM must be fitted first');
    }
    return this.quantization_errors_[this.quantization_errors_.length - 1];
  }
  
  /**
   * Calculate topographic error.
   * Proportion of samples whose BMU and second BMU are not neighbors.
   */
  async topographic_error(X?: DataMatrix): Promise<number> {
    if (!this.weights_) {
      throw new Error('SOM must be fitted first');
    }

    if (!X) {
      throw new Error('Input data required for topographic error calculation');
    }

    const x_tensor = is_tensor(X) ? X as tf.Tensor2D : tf.tensor2d(X);

    try {
      const { grid_width, grid_height, topology } = this.params;
      const [n_samples, n_features] = x_tensor.shape;
      const total_neurons = grid_height * grid_width;

      // Compute distance matrix [n_samples, total_neurons] in one batch
      const { bmu1_coords, bmu2_coords } = tf.tidy(() => {
        const weights_flat = this.weights_!.reshape([total_neurons, n_features]);
        const samples_norm = x_tensor.square().sum(1, true);
        const weights_norm = weights_flat.square().sum(1, true).transpose();
        const dot_product = tf.mat_mul(x_tensor, weights_flat, false, true);
        const distances = samples_norm.add(weights_norm).sub(dot_product.mul(2));

        // First BMU: argmin of distances
        const bmu1_indices = distances.argMin(1);

        // Second BMU: mask first BMU with large value, then argmin again
        const one_hot = tf.one_hot(bmu1_indices, total_neurons);
        const masked_distances = distances.add(one_hot.mul(1e30));
        const bmu2_indices = masked_distances.argMin(1);

        // Convert flat indices to grid coordinates
        const bmu1_rows = bmu1_indices.div(grid_width).floor();
        const bmu1_cols = bmu1_indices.mod(grid_width);
        const bmu2_rows = bmu2_indices.div(grid_width).floor();
        const bmu2_cols = bmu2_indices.mod(grid_width);

        return {
          bmu1_coords: tf.stack([bmu1_rows, bmu1_cols], 1) as tf.Tensor2D,
          bmu2_coords: tf.stack([bmu2_rows, bmu2_cols], 1) as tf.Tensor2D,
        };
      });

      // Read coordinates and check neighbors in plain JS
      const bmu1_data = bmu1_coords.dataSync();
      const bmu2_data = bmu2_coords.dataSync();
      bmu1_coords.dispose();
      bmu2_coords.dispose();

      let errors = 0;
      for (let i = 0; i < n_samples; i++) {
        const is_neighbor = this.are_neighbors(
          bmu1_data[i * 2], bmu1_data[i * 2 + 1],
          bmu2_data[i * 2], bmu2_data[i * 2 + 1],
          grid_height, grid_width, topology!,
        );
        if (!is_neighbor) {
          errors++;
        }
      }

      return errors / n_samples;
    } finally {
      if (!is_tensor(X)) {
        x_tensor.dispose();
      }
    }
  }
  
  /**
   * Check if two neurons are neighbors in the grid.
   */
  private are_neighbors(
    row1: number, col1: number,
    row2: number, col2: number,
    grid_height: number, grid_width: number,
    topology: SOMTopology
  ): boolean {
    if (topology === 'rectangular') {
      // 8-connected rectangular grid (consistent with get_u_matrix and get_neighbors)
      const row_diff = Math.abs(row1 - row2);
      const col_diff = Math.abs(col1 - col2);

      return row_diff <= 1 && col_diff <= 1 && (row_diff + col_diff > 0);
    } else {
      // Hexagonal topology (6-connected)
      const row_diff = row2 - row1;
      const col_diff = col2 - col1;
      
      // Even rows have different neighbor offsets than odd rows
      const even_row = row1 % 2 === 0;
      
      // Check all 6 possible hexagonal neighbors
      const hex_neighbors = even_row ? [
        [-1, -1], [-1, 0],  // Top-left, top-right
        [0, -1], [0, 1],    // Left, right
        [1, -1], [1, 0]     // Bottom-left, bottom-right
      ] : [
        [-1, 0], [-1, 1],   // Top-left, top-right
        [0, -1], [0, 1],    // Left, right
        [1, 0], [1, 1]      // Bottom-left, bottom-right
      ];
      
      return hex_neighbors.some(([dr, dc]) => dr === row_diff && dc === col_diff);
    }
  }
  
  /**
   * Get total samples learned (for online learning).
   */
  get_total_samples_learned(): number {
    return this.total_samples_learned_;
  }
  
  /**
   * Save SOM state for persistence.
   */
  save_state(): SOMState {
    if (!this.weights_) {
      throw new Error('SOM must be fitted first');
    }
    
    return {
      weights: this.weights_.arraySync(),
      total_samples: this.total_samples_learned_,
      current_epoch: this.current_epoch_,
      grid_width: this.params.grid_width,
      grid_height: this.params.grid_height,
      params: this.params,
    };
  }
  
  /**
   * Load SOM state from saved data.
   */
  load_state(state: SOMState): void {
    this.weights_?.dispose();
    this.weights_ = tf.tensor3d(state.weights);
    this.total_samples_learned_ = state.total_samples;
    this.current_epoch_ = state.current_epoch;
    
    // Reinitialize schedulers if params changed
    if (state.params) {
      this.initialize_schedulers();
    }
  }
  
  /**
   * Save model state to JSON string.
   * Can be saved to file or transmitted over network.
   */
  async save_to_json(): Promise<string> {
    if (!this.weights_) {
      throw new Error('SOM must be fitted before saving');
    }
    
    const model_data = {
      weights: await this.weights_.array(),
      metadata: {
        params: this.params,
        total_samples_learned: this.total_samples_learned_,
        current_epoch: this.current_epoch_,
        quantization_errors: this.quantization_errors_,
      }
    };
    
    return JSON.stringify(model_data, null, 2);
  }
  
  /**
   * Load model from JSON string.
   * 
   * @param json JSON string containing model data
   */
  async load_from_json(json: string): Promise<void> {
    const model_data = JSON.parse(json);
    
    // Dispose existing weights
    this.weights_?.dispose();
    
    // Load weights
    this.weights_ = tf.tensor3d(model_data.weights);
    
    // Load metadata
    if (model_data.metadata) {
      // Create new params object instead of modifying readonly
      const loaded_params = model_data.metadata.params;
      if (loaded_params) {
        // Reconstruct the object with new params to maintain immutability
        Object.defineProperty(this, 'params', {
          value: loaded_params,
          writable: false,
          configurable: true
        });
      }
      this.total_samples_learned_ = model_data.metadata.totalSamplesLearned || 0;
      this.current_epoch_ = model_data.metadata.currentEpoch || 0;
      this.quantization_errors_ = model_data.metadata.quantizationErrors || [];
    }
    
    // Reinitialize schedulers and distance matrix
    this.initialize_schedulers();
    this.grid_distance_matrix_ = create_grid_distance_matrix(
      this.params.grid_height,
      this.params.grid_width,
      this.params.topology!
    );
  }
  
  /**
   * Enable streaming mode for continuous learning.
   * Configures the SOM for online learning with optimal settings.
   */
  enable_streaming_mode(batch_size?: number): void {
    Object.defineProperty(this, 'params', {
      value: {
        ...this.params,
        online_mode: true,
        mini_batch_size: batch_size || SOM.DEFAULT_MINI_BATCH_SIZE,
      },
      writable: false,
      configurable: true,
    });
    
    // Adjust schedulers for continuous learning
    // Use slower decay for streaming scenarios
    const { grid_width, grid_height } = this.params;
    const initial_learning_rate = typeof this.params.learning_rate === 'number' 
      ? this.params.learning_rate 
      : SOM.DEFAULT_LEARNING_RATE;
    
    this.learning_rate_scheduler_ = (_epoch: number, _total_epochs: number) => {
      // Slower decay for streaming
      const virtual_epoch = this.total_samples_learned_ / 1000; // Adjust scale
      return initial_learning_rate * Math.exp(-virtual_epoch / 100);
    };
    
    const initial_radius = Math.max(grid_width, grid_height) / 2;
    this.radius_scheduler_ = (_epoch: number, _total_epochs: number) => {
      const virtual_epoch = this.total_samples_learned_ / 1000;
      return Math.max(1, initial_radius * Math.exp(-virtual_epoch / 50));
    };
  }
  
  /**
   * Process a stream of data samples.
   * Automatically manages batch accumulation and training.
   * 
   * @param sample Single sample or small batch
   * @param auto_train Whether to train immediately or accumulate
   */
  async process_stream(
    sample: DataMatrix,
    auto_train: boolean = true
  ): Promise<void> {
    if (!this.params.online_mode) {
      this.enable_streaming_mode();
    }
    
    const x_tensor = is_tensor(sample) ? sample as tf.Tensor2D : tf.tensor2d(sample);
    
    try {
      if (auto_train) {
        await this.partial_fit(x_tensor);
      } else {
        throw new Error(
          'processStream with autoTrain=false is not supported. ' +
          'Use autoTrain=true or call partialFit() directly.',
        );
      }
    } finally {
      if (!is_tensor(sample)) {
        x_tensor.dispose();
      }
    }
  }
  
  /**
   * Get streaming statistics.
   */
  get_streaming_stats(): {
    total_samples: number;
    virtual_epoch: number;
    current_learning_rate: number;
    current_radius: number;
    latest_quantization_error: number;
  } {
    const batch_size = this.last_batch_size_ || this.params.mini_batch_size || 1;
    const virtual_epoch = Math.floor(this.total_samples_learned_ / batch_size);
    
    return {
      total_samples: this.total_samples_learned_,
      virtual_epoch,
      current_learning_rate: this.learning_rate_scheduler_?.(
        virtual_epoch,
        this.params.num_epochs || 100
      ) || 0,
      current_radius: this.radius_scheduler_?.(
        virtual_epoch,
        this.params.num_epochs || 100
      ) || 0,
      latest_quantization_error: this.quantization_errors_.length > 0
        ? this.quantization_errors_[this.quantization_errors_.length - 1]
        : 0,
    };
  }
  
  /**
   * Create a shuffled array of indices [0..n-1] using Fisher-Yates.
   */
  private shuffle_indices(n: number, rng: RandomStream): Int32Array {
    const indices = new Int32Array(n);
    for (let i = 0; i < n; i++) indices[i] = i;
    for (let i = n - 1; i > 0; i--) {
      const j = rng.rand_int(i + 1);
      const tmp = indices[i];
      indices[i] = indices[j];
      indices[j] = tmp;
    }
    return indices;
  }

  /**
   * Releases all GPU/WebGL memory held by this SOM instance.
   *
   * After calling `dispose()`, the SOM instance must not be used for any
   * further operations (fit, predict, cluster, get_weights, get_u_matrix, etc.).
   *
   * Values previously returned by {@link get_weights} (plain `number[][][]`
   * arrays) remain valid after disposal. However, any `tf.Tensor` values
   * previously returned by {@link get_u_matrix} that have not yet been disposed
   * by the caller are unaffected — the caller still owns those tensors and
   * must dispose them separately.
   *
   * Calling `dispose()` multiple times is safe (idempotent).
   */
  dispose(): void {
    this.weights_?.dispose();
    this.weights_ = null;
    this.grid_distance_matrix_?.dispose();
    this.grid_distance_matrix_ = null;
    this.bmus_?.dispose();
    this.bmus_ = null;
    this.labels_ = null;
  }
}