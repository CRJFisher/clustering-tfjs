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

  public weights_: tf.Tensor3D | null = null;
  public labels_: number[] | null = null;
  public bmus_: tf.Tensor2D | null = null;

  private grid_distance_matrix_: tf.Tensor2D | null = null;
  private learning_rate_scheduler_: DecayFunction | null = null;
  private radius_scheduler_: DecayFunction | null = null;
  private total_samples_learned_: number = 0;
  private last_batch_size_: number = 0;
  private current_epoch_: number = 0;
  private quantization_errors_: number[] = [];

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
  
  private validate_and_complete_params(params: SOMParams): SOMParams {
    if (!params.grid_width || params.grid_width < 1) {
      throw new Error('grid_width must be >= 1');
    }
    if (!params.grid_height || params.grid_height < 1) {
      throw new Error('grid_height must be >= 1');
    }
    if (params.initial_weights) {
      this.validate_initial_weights_shape(params.initial_weights, params.grid_height, params.grid_width);
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
  
  private validate_initial_weights_shape(
    initial_weights: number[][][],
    grid_height: number,
    grid_width: number,
  ): void {
    if (initial_weights.length !== grid_height) {
      throw new Error(
        `initial_weights must have grid_height (${grid_height}) rows, got ${initial_weights.length}`,
      );
    }
    let n_features = -1;
    for (const row of initial_weights) {
      if (row.length !== grid_width) {
        throw new Error(
          `initial_weights rows must have grid_width (${grid_width}) entries, got ${row.length}`,
        );
      }
      for (const weight of row) {
        if (n_features === -1) {
          n_features = weight.length;
        }
        if (weight.length !== n_features) {
          throw new Error('initial_weights neuron vectors must all share the same length');
        }
      }
    }
    if (n_features < 1) {
      throw new Error('initial_weights neuron vectors must have at least one feature');
    }
  }

  private make_initial_weights(X: tf.Tensor2D, n_features: number): tf.Tensor3D {
    if (this.params.initial_weights) {
      const injected_features = this.params.initial_weights[0][0].length;
      if (injected_features !== n_features) {
        throw new Error(
          `initial_weights feature dimension (${injected_features}) does not match data (${n_features})`,
        );
      }
      return tf.tensor3d(this.params.initial_weights);
    }
    return initialize_weights(
      X,
      this.params.grid_height,
      this.params.grid_width,
      this.params.initialization!,
      this.params.random_state,
    );
  }

  private initialize_schedulers(): void {
    const { grid_width, grid_height, num_epochs, learning_rate, radius } = this.params;

    if (typeof learning_rate === 'function') {
      this.learning_rate_scheduler_ = learning_rate;
    } else {
      this.learning_rate_scheduler_ = create_decay_scheduler(
        learning_rate as number,
        'exponential',
        num_epochs!
      );
    }

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
      const initial_radius = Math.max(grid_width, grid_height) / 2;
      this.radius_scheduler_ = create_decay_scheduler(
        initial_radius,
        'exponential',
        num_epochs!,
        1 // final radius
      );
    }
  }
  
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
  
  private async fit_tensor(X: tf.Tensor2D): Promise<void> {
    const {
      grid_width,
      grid_height,
      topology,
      num_epochs,
      random_state,
      tol
    } = this.params;
    
    const [n_samples, n_features] = X.shape;

    if (!this.weights_) {
      this.weights_ = this.make_initial_weights(X, n_features);
    }

    if (!this.grid_distance_matrix_) {
      this.grid_distance_matrix_ = create_grid_distance_matrix(
        grid_height,
        grid_width,
        topology!
      );
    }
    
    let prev_quantization_error = Infinity;
    this.quantization_errors_ = [];
    const rng = make_random_stream(random_state);

    for (let epoch = 0; epoch < num_epochs!; epoch++) {
      this.current_epoch_ = epoch;

      const current_learning_rate = this.learning_rate_scheduler_!(epoch, num_epochs!);
      const current_radius = this.radius_scheduler_!(epoch, num_epochs!);

      validate_neighborhood_params(current_radius, grid_height, grid_width);

      // Shuffle data each epoch to avoid order-dependent bias
      const shuffled_indices = this.shuffle_indices(n_samples, rng);
      const indices_tensor = tf.tensor1d(shuffled_indices, 'int32');
      const shuffled_x = tf.gather(X, indices_tensor) as tf.Tensor2D;
      indices_tensor.dispose();

      const { quantization_error } = await this.train_epoch(
        shuffled_x,
        current_learning_rate,
        current_radius
      );

      shuffled_x.dispose();

      this.quantization_errors_.push(quantization_error);

      if (Math.abs(prev_quantization_error - quantization_error) < tol!) {
        break;
      }

      prev_quantization_error = quantization_error;

      if (this.params.online_mode) {
        this.total_samples_learned_ += n_samples;
      }
    }
    
    await this.compute_final_labels(X);
  }

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
      
      const bmus = find_bmu_batch(batch_x, this.weights_!);

      const bmu_indices = tf.tidy(() => {
        const bmus_data = bmus.arraySync();
        const indices = bmus_data.map(([row, col]) => 
          row * this.params.grid_width + col
        );
        return tf.tensor1d(indices, 'int32');
      });
      
      const influence = compute_neighborhood_influence_batch(
        bmu_indices,
        this.grid_distance_matrix_!,
        radius,
        neighborhood!
      );
      
      this.update_weights(batch_x, influence, learning_rate);

      const distances = compute_bmu_distances(batch_x, this.weights_!, bmus);
      const batch_error = distances.mean().arraySync() as number;
      total_quantization_error += batch_error * (end_idx - i);
      samples_processed += (end_idx - i);
      
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
  
  private update_weights(
    samples: tf.Tensor2D,
    influence: tf.Tensor2D,
    learning_rate: number
  ): void {
    tf.tidy(() => {
      const [_n_samples, n_features] = samples.shape;
      const [grid_height, grid_width, _n_features_weight] = this.weights_!.shape;
      const total_neurons = grid_height * grid_width;
      
      const weights_flat = this.weights_!.reshape([total_neurons, n_features]);
      
      // Batch SOM update for each neuron j:
      // Δw_j = lr * Σ_i(h_ij * (x_i - w_j)) / Σ_i(h_ij)
      // Normalizes by sum of influences to make updates independent of batch size

      const samples_expanded = samples.expandDims(1); // [n_samples, 1, n_features]
      const weights_expanded = weights_flat.expandDims(0); // [1, total_neurons, n_features]

      const diff = samples_expanded.sub(weights_expanded); // [n_samples, total_neurons, n_features]

      const influence_expanded = influence.expandDims(2); // [n_samples, total_neurons, 1]
      const weighted_diff = diff.mul(influence_expanded);

      const total_update = weighted_diff.sum(0); // [total_neurons, n_features]

      // Normalize by sum of influences per neuron (sign-preserving for mexican_hat)
      const influence_sum = influence.sum(0); // [total_neurons]
      const epsilon = 1e-8;
      const abs_influence_sum = influence_sum.abs();
      const influence_sum_safe = tf.where(
        abs_influence_sum.greater(epsilon),
        influence_sum,
        tf.fill(influence_sum.shape, epsilon)
      );
      const normalized_update = total_update.div(influence_sum_safe.expandDims(1));

      const new_weights_flat = weights_flat.add(normalized_update.mul(learning_rate));

      const new_weights = new_weights_flat.reshape([grid_height, grid_width, n_features]) as tf.Tensor3D;
      
      // Update weights in place (keep new_weights from being disposed by tidy)
      this.weights_!.dispose();
      this.weights_ = tf.keep(new_weights);
    });
  }
  
  private async compute_final_labels(X: tf.Tensor2D): Promise<void> {
    this.bmus_ = find_bmu_batch(X, this.weights_!);

    const bmus_data = await this.bmus_.array();
    const labels = bmus_data.map(([row, col]) => 
      row * this.params.grid_width + col
    );
    
    this.labels_ = labels;
  }
  
  async fit_predict(X: DataMatrix): Promise<number[]> {
    await this.fit(X);
    return this.labels_!;
  }
  
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
   * SOM neurons outnumber the desired clusters (grid_width * grid_height >> n_clusters),
   * so raw BMU indices are not useful as cluster assignments. Applies agglomerative
   * clustering on the trained weight vectors to group neurons into `n_clusters`
   * macro-clusters, then maps each data point's BMU to its macro-cluster label.
   *
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
      throw new Error('n_clusters must be a positive integer (>= 1).');
    }
    if (n_clusters > total_neurons) {
      throw new Error(
        `n_clusters (${n_clusters}) exceeds total number of neurons (${total_neurons}). Maximum is grid_width * grid_height.`
      );
    }

      const weights_data = this.weights_.arraySync();
    const neuron_vectors: number[][] = [];
    for (let row = 0; row < grid_height; row++) {
      for (let col = 0; col < grid_width; col++) {
        neuron_vectors.push(weights_data[row][col]);
      }
    }

    const agglo = new AgglomerativeClustering({
      n_clusters,
      linkage: options?.linkage ?? 'ward',
      metric: options?.metric ?? 'euclidean',
    });

    await agglo.fit(neuron_vectors);
    const neuron_labels = agglo.labels_!;

    return this.labels_.map(bmu_index => neuron_labels[bmu_index]);
  }

  /**
   * On the first call (when no weights exist), the input dimensionality establishes
   * the expected feature count. Subsequent calls must match that dimension.
   *
   * @throws Error if online_mode is not enabled.
   * @throws Error if n_features does not match the feature dimensionality of existing weights.
   */
  async partial_fit(X: DataMatrix): Promise<void> {
    if (!this.params.online_mode) {
      throw new Error('partial_fit requires online_mode to be enabled');
    }
    
    const x_tensor = is_tensor(X) ? X as tf.Tensor2D : tf.tensor2d(X);
    
    try {
      const [n_samples] = x_tensor.shape;
      this.last_batch_size_ = n_samples;

      if (!this.weights_) {
        this.weights_ = this.make_initial_weights(x_tensor, x_tensor.shape[1]);

        this.grid_distance_matrix_ = create_grid_distance_matrix(
          this.params.grid_height,
          this.params.grid_width,
          this.params.topology!
        );

        this.initialize_schedulers();
      } else {
        const expected_features = this.weights_.shape[2];
        const actual_features = x_tensor.shape[1];
        if (actual_features !== expected_features) {
          throw new Error(
            `Feature dimension mismatch: expected ${expected_features} features to match prior fit, but got ${actual_features}`
          );
        }
      }

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
      
      await this.train_epoch(x_tensor, current_learning_rate, current_radius);

      this.total_samples_learned_ += n_samples;

      await this.compute_final_labels(x_tensor);
    } finally {
      if (!is_tensor(X)) {
        x_tensor.dispose();
      }
    }
  }
  
  /**
   * The array has shape `[grid_height][grid_width][n_features]`; each element
   * `weights[row][col]` is the codebook vector for that neuron.
   *
   * Snapshot (deep copy): mutating it won't affect the SOM, and {@link dispose}
   * won't invalidate it.
   *
   * @throws Error if the SOM has not been fitted yet.
   */
  get_weights(): number[][][] {
    if (!this.weights_) {
      throw new Error('SOM must be fitted first');
    }
    return this.weights_.arraySync();
  }
  
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
          
          let neighbors: number[][];
          
          if (topology === 'rectangular') {
            neighbors = [
              [i - 1, j], [i + 1, j],
              [i, j - 1], [i, j + 1],
              [i - 1, j - 1], [i - 1, j + 1],
              [i + 1, j - 1], [i + 1, j + 1]
            ];
          } else {
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
  
  quantization_error(): number {
    if (this.quantization_errors_.length === 0) {
      throw new Error('SOM must be fitted first');
    }
    return this.quantization_errors_[this.quantization_errors_.length - 1];
  }
  
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

      const { bmu1_coords, bmu2_coords } = tf.tidy(() => {
        const weights_flat = this.weights_!.reshape([total_neurons, n_features]);
        const samples_norm = x_tensor.square().sum(1, true);
        const weights_norm = weights_flat.square().sum(1, true).transpose();
        const dot_product = tf.mat_mul(x_tensor, weights_flat, false, true);
        const distances = samples_norm.add(weights_norm).sub(dot_product.mul(2));

        const bmu1_indices = distances.argMin(1);

        // Mask first BMU with a large value so the second argmin finds a different neuron.
        const one_hot = tf.one_hot(bmu1_indices, total_neurons);
        const masked_distances = distances.add(one_hot.mul(1e30));
        const bmu2_indices = masked_distances.argMin(1);

        const bmu1_rows = bmu1_indices.div(grid_width).floor();
        const bmu1_cols = bmu1_indices.mod(grid_width);
        const bmu2_rows = bmu2_indices.div(grid_width).floor();
        const bmu2_cols = bmu2_indices.mod(grid_width);

        return {
          bmu1_coords: tf.stack([bmu1_rows, bmu1_cols], 1) as tf.Tensor2D,
          bmu2_coords: tf.stack([bmu2_rows, bmu2_cols], 1) as tf.Tensor2D,
        };
      });

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
  
  private are_neighbors(
    row1: number, col1: number,
    row2: number, col2: number,
    grid_height: number, grid_width: number,
    topology: SOMTopology
  ): boolean {
    if (topology === 'rectangular') {
      // Neighbour definition must stay consistent with get_u_matrix and get_neighbors.
      const row_diff = Math.abs(row1 - row2);
      const col_diff = Math.abs(col1 - col2);

      return row_diff <= 1 && col_diff <= 1 && (row_diff + col_diff > 0);
    } else {
      const row_diff = row2 - row1;
      const col_diff = col2 - col1;

      // Hex neighbour offsets differ between even and odd rows.
      const even_row = row1 % 2 === 0;

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
  
  get_total_samples_learned(): number {
    return this.total_samples_learned_;
  }
  
  save_state(): SOMState {
    if (!this.weights_) {
      throw new Error('SOM must be fitted first');
    }
    
    // Trained weights are persisted separately, so the injected initial_weights
    // grid carries no value in a snapshot and is dropped to avoid duplicating it.
    const { initial_weights: _initial_weights, ...persisted_params } = this.params;
    return {
      weights: this.weights_.arraySync(),
      total_samples: this.total_samples_learned_,
      current_epoch: this.current_epoch_,
      grid_width: this.params.grid_width,
      grid_height: this.params.grid_height,
      params: persisted_params,
    };
  }
  
  load_state(state: SOMState): void {
    this.weights_?.dispose();
    this.weights_ = tf.tensor3d(state.weights);
    this.total_samples_learned_ = state.total_samples;
    this.current_epoch_ = state.current_epoch;
    
    if (state.params) {
      this.initialize_schedulers();
    }
  }
  
  async save_to_json(): Promise<string> {
    if (!this.weights_) {
      throw new Error('SOM must be fitted before saving');
    }
    
    const { initial_weights: _initial_weights, ...persisted_params } = this.params;
    const model_data = {
      weights: await this.weights_.array(),
      metadata: {
        params: persisted_params,
        total_samples_learned: this.total_samples_learned_,
        current_epoch: this.current_epoch_,
        quantization_errors: this.quantization_errors_,
      }
    };
    
    return JSON.stringify(model_data, null, 2);
  }
  
  async load_from_json(json: string): Promise<void> {
    const model_data = JSON.parse(json);

    this.weights_?.dispose();

    this.weights_ = tf.tensor3d(model_data.weights);

    if (model_data.metadata) {
      const loaded_params = model_data.metadata.params;
      if (loaded_params) {
        Object.defineProperty(this, 'params', {
          value: loaded_params,
          writable: false,
          configurable: true
        });
      }
      this.total_samples_learned_ = model_data.metadata.total_samples_learned || 0;
      this.current_epoch_ = model_data.metadata.current_epoch || 0;
      this.quantization_errors_ = model_data.metadata.quantization_errors || [];
    }
    
    this.initialize_schedulers();
    this.grid_distance_matrix_ = create_grid_distance_matrix(
      this.params.grid_height,
      this.params.grid_width,
      this.params.topology!
    );
  }
  
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
    
    // Use slower decay for streaming: avoids catastrophic forgetting on long streams.
    const { grid_width, grid_height } = this.params;
    const initial_learning_rate = typeof this.params.learning_rate === 'number' 
      ? this.params.learning_rate 
      : SOM.DEFAULT_LEARNING_RATE;
    
    this.learning_rate_scheduler_ = (_epoch: number, _total_epochs: number) => {
      const virtual_epoch = this.total_samples_learned_ / 1000;
      return initial_learning_rate * Math.exp(-virtual_epoch / 100);
    };
    
    const initial_radius = Math.max(grid_width, grid_height) / 2;
    this.radius_scheduler_ = (_epoch: number, _total_epochs: number) => {
      const virtual_epoch = this.total_samples_learned_ / 1000;
      return Math.max(1, initial_radius * Math.exp(-virtual_epoch / 50));
    };
  }
  
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
          'process_stream with auto_train=false is not supported. ' +
          'Use auto_train=true or call partial_fit() directly.',
        );
      }
    } finally {
      if (!is_tensor(sample)) {
        x_tensor.dispose();
      }
    }
  }
  
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
  
  // Fisher-Yates shuffle
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