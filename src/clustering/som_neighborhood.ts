import * as tf from '../backend/adapter';
import type { SOMTopology, SOMInitialization, SOMNeighborhood, DecayFunction } from './types';
import { power_iteration_eig } from '../decomposition/pca';

/**
 * Grid coordinate utilities for SOM topology management.
 */

/**
 * Convert 2D grid coordinates to 1D index.
 */
export function grid_to_index(row: number, col: number, grid_width: number): number {
  return row * grid_width + col;
}

/**
 * Convert 1D index to 2D grid coordinates.
 */
export function index_to_grid(index: number, grid_width: number): [number, number] {
  const row = Math.floor(index / grid_width);
  const col = index % grid_width;
  return [row, col];
}

/**
 * Calculate distance between two grid positions based on topology.
 * 
 * @param pos1 First position [row, col]
 * @param pos2 Second position [row, col]
 * @param topology Grid topology type
 * @returns Euclidean distance between positions
 */
export function grid_distance(
  pos1: [number, number],
  pos2: [number, number],
  topology: SOMTopology
): number {
  const [r1, c1] = pos1;
  const [r2, c2] = pos2;
  
  if (topology === 'rectangular') {
    // Simple Euclidean distance for rectangular grid
    return Math.sqrt((r1 - r2) ** 2 + (c1 - c2) ** 2);
  } else {
    // Hexagonal grid distance calculation
    // In hexagonal topology, even rows are offset by 0.5
    const x1 = c1 + (r1 % 2) * 0.5;
    const y1 = r1 * Math.sqrt(3) / 2;
    const x2 = c2 + (r2 % 2) * 0.5;
    const y2 = r2 * Math.sqrt(3) / 2;
    
    return Math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2);
  }
}

/**
 * Get neighbors of a grid position based on topology.
 * 
 * @param row Row position
 * @param col Column position
 * @param grid_height Grid height
 * @param grid_width Grid width
 * @param topology Grid topology
 * @returns Array of neighbor positions [row, col]
 */
export function get_neighbors(
  row: number,
  col: number,
  grid_height: number,
  grid_width: number,
  topology: SOMTopology
): Array<[number, number]> {
  const neighbors: Array<[number, number]> = [];
  
  if (topology === 'rectangular') {
    // 8-connected neighborhood for rectangular grid
    const deltas = [
      [-1, -1], [-1, 0], [-1, 1],
      [0, -1],           [0, 1],
      [1, -1],  [1, 0],  [1, 1]
    ];
    
    for (const [dr, dc] of deltas) {
      const new_row = row + dr;
      const new_col = col + dc;
      if (new_row >= 0 && new_row < grid_height && 
          new_col >= 0 && new_col < grid_width) {
        neighbors.push([new_row, new_col]);
      }
    }
  } else {
    // Hexagonal grid neighbors (6-connected)
    // Even rows have different neighbor offsets than odd rows
    const even_row_deltas = [
      [-1, -1], [-1, 0],
      [0, -1],  [0, 1],
      [1, -1],  [1, 0]
    ];
    const odd_row_deltas = [
      [-1, 0],  [-1, 1],
      [0, -1],  [0, 1],
      [1, 0],   [1, 1]
    ];
    
    const deltas = row % 2 === 0 ? even_row_deltas : odd_row_deltas;
    
    for (const [dr, dc] of deltas) {
      const new_row = row + dr;
      const new_col = col + dc;
      if (new_row >= 0 && new_row < grid_height && 
          new_col >= 0 && new_col < grid_width) {
        neighbors.push([new_row, new_col]);
      }
    }
  }
  
  return neighbors;
}

/**
 * Initialize SOM weights based on the specified strategy.
 * 
 * @param X Input data tensor [n_samples, n_features]
 * @param grid_height Height of the SOM grid
 * @param grid_width Width of the SOM grid
 * @param initialization Initialization strategy
 * @param random_seed Random seed for reproducibility
 * @returns Weight tensor [grid_height, grid_width, n_features]
 */
export function initialize_weights(
  X: tf.Tensor2D,
  grid_height: number,
  grid_width: number,
  initialization: SOMInitialization,
  random_seed?: number
): tf.Tensor3D {
  return tf.tidy(() => {
    const [_n_samples, n_features] = X.shape;
    
    switch (initialization) {
      case 'random': {
        // Random initialization from data range
        const x_min = X.min(0);
        const x_max = X.max(0);
        const range = x_max.sub(x_min);
        
        // Create random weights within data range
        const random_weights = tf.random_uniform(
          [grid_height, grid_width, n_features],
          0,
          1,
          undefined,
          random_seed
        );
        
        // Scale to data range
        return random_weights
          .mul(range.reshape([1, 1, n_features]))
          .add(x_min.reshape([1, 1, n_features]));
      }
      
      case 'linear': {
        // Linear initialization: span 2D grid along first two principal components
        // Following MiniSom's pca_weights_init approach
        const [n_samples_lin, n_features_lin] = X.shape;

        if (n_samples_lin < 2 || n_features_lin < 2) {
          return initialize_weights(X, grid_height, grid_width, 'random', random_seed);
        }

        // Center the data
        const mean = X.mean(0, true); // [1, n_features]
        const centered = X.sub(mean);

        // Compute covariance matrix
        const cov = tf.mat_mul(centered, centered, true, false).div(n_samples_lin - 1);

        // Get first 2 principal components via power iteration
        const n_comps = Math.min(2, n_features_lin);
        const components = compute_principal_components(cov as tf.Tensor2D, n_comps, random_seed);

        // Project data onto PCs to determine scale (unbiased variance, consistent with covariance)
        const projections = tf.mat_mul(centered, components, false, true); // [n_samples, n_comps]
        const proj_var = projections.square().sum(0).div(n_samples_lin - 1); // unbiased (centered data has mean 0)
        const proj_std = proj_var.sqrt();
        const proj_std_data = proj_std.dataSync();

        // Scale eigenvectors by projection standard deviations
        const components_data = components.arraySync() as number[][];
        const scaled_pc1 = components_data[0].map(v => v * proj_std_data[0]);
        const scaled_pc2 = n_comps > 1
          ? components_data[1].map(v => v * proj_std_data[1])
          : new Array(n_features_lin).fill(0);

        const mean_data = mean.squeeze().dataSync();

        // Build weights: mean + c1 * scaled_pc1 + c2 * scaled_pc2
        // c1, c2 range from -1 to 1 (MiniSom convention)
        const weights: number[][][] = [];
        for (let i = 0; i < grid_height; i++) {
          const row_weights: number[][] = [];
          const c1 = grid_height > 1 ? -1 + 2 * i / (grid_height - 1) : 0;
          for (let j = 0; j < grid_width; j++) {
            const c2 = grid_width > 1 ? -1 + 2 * j / (grid_width - 1) : 0;
            const weight: number[] = [];
            for (let f = 0; f < n_features_lin; f++) {
              weight.push(mean_data[f] + c1 * scaled_pc1[f] + c2 * scaled_pc2[f]);
            }
            row_weights.push(weight);
          }
          weights.push(row_weights);
        }

        // Cleanup
        mean.dispose();
        centered.dispose();
        cov.dispose();
        components.dispose();
        projections.dispose();
        proj_var.dispose();
        proj_std.dispose();

        return tf.tensor3d(weights);
      }
      
      case 'pca': {
        // PCA-based initialization - place weights along principal components
        const [_n_samples, n_features] = X.shape;
        
        if (_n_samples < 2) {
          // Fall back to random initialization if not enough samples
          return initialize_weights(X, grid_height, grid_width, 'random', random_seed);
        }
        
        // Center the data
        const mean = X.mean(0, true);
        const centered = X.sub(mean);
        
        // Compute covariance matrix
        const cov = tf.mat_mul(centered, centered, true, false).div(_n_samples - 1);
        
        // Get principal components
        const n_comps = Math.min(2, n_features);
        const components = compute_principal_components(cov as tf.Tensor2D, n_comps, random_seed);
        
        // Project data onto principal components
        const projections = tf.mat_mul(centered, components, false, true);
        const pc_min = projections.min(0);
        const pc_max = projections.max(0);
        const pc_range = pc_max.sub(pc_min);
        
        // Create grid in PC space and transform back
        const weights: number[][][] = [];
        
        for (let i = 0; i < grid_height; i++) {
          const row_weights: number[][] = [];
          for (let j = 0; j < grid_width; j++) {
            // Map grid position to PC space
            const alpha = grid_height > 1 ? i / (grid_height - 1) : 0.5;
            const beta = grid_width > 1 ? j / (grid_width - 1) : 0.5;
            
            // Create PC coordinates
            const pc_coords = tf.zeros([n_comps]);
            const pc_buffer = pc_coords.bufferSync();
            pc_buffer.set(pc_min.dataSync()[0] + pc_range.dataSync()[0] * alpha, 0);
            if (n_comps > 1) {
              pc_buffer.set(pc_min.dataSync()[1] + pc_range.dataSync()[1] * beta, 1);
            }
            
            // Transform back to original space
            const weight = tf.tidy(() => {
              const reconstructed = tf.mat_mul(
                pc_buffer.toTensor().expandDims(0),
                components
              );
              return reconstructed.squeeze().add(mean);
            });
            
            row_weights.push(Array.from(weight.dataSync()));
            weight.dispose();
            pc_coords.dispose();
          }
          weights.push(row_weights);
        }
        
        // Cleanup
        cov.dispose();
        components.dispose();
        projections.dispose();
        pc_min.dispose();
        pc_max.dispose();
        pc_range.dispose();
        centered.dispose();
        mean.dispose();
        
        return tf.tensor3d(weights);
      }
      
      default:
        throw new Error(`Unknown initialization strategy: ${initialization}`);
    }
  });
}

/**
 * Compute principal components of a covariance matrix.
 *
 * Sources the leading components from the shared power-iteration eigensolver in
 * `src/decomposition/pca.ts`, so SOM `'linear'`/`'pca'` initialization and the
 * public `PCA` estimator use one numerically identical computation.
 *
 * @param cov_matrix Covariance matrix [n_features, n_features]
 * @param n_components Number of components to extract
 * @param random_state Seed for deterministic power-iteration initialization
 * @returns Principal components [n_components, n_features]
 */
function compute_principal_components(
  cov_matrix: tf.Tensor2D,
  n_components: number,
  random_state?: number,
): tf.Tensor2D {
  const cov = cov_matrix.arraySync() as number[][];
  const { components } = power_iteration_eig(cov, n_components, random_state);
  return tf.tensor2d(components, [components.length, cov.length], 'float32');
}

/**
 * Create a distance matrix between all grid positions.
 * Used for neighborhood calculations.
 * 
 * @param grid_height Grid height
 * @param grid_width Grid width
 * @param topology Grid topology
 * @returns Distance matrix [total_neurons, total_neurons]
 */
export function create_grid_distance_matrix(
  grid_height: number,
  grid_width: number,
  topology: SOMTopology
): tf.Tensor2D {
  const total_neurons = grid_height * grid_width;
  const distances: number[][] = [];
  
  for (let i = 0; i < total_neurons; i++) {
    const pos1 = index_to_grid(i, grid_width);
    const row: number[] = [];
    
    for (let j = 0; j < total_neurons; j++) {
      const pos2 = index_to_grid(j, grid_width);
      row.push(grid_distance(pos1, pos2, topology));
    }
    distances.push(row);
  }
  
  return tf.tensor2d(distances);
}

/**
 * Generate grid coordinates for visualization.
 * 
 * @param grid_height Grid height
 * @param grid_width Grid width
 * @param topology Grid topology
 * @returns Coordinates tensor [total_neurons, 2] with (x, y) positions
 */
export function get_grid_coordinates(
  grid_height: number,
  grid_width: number,
  topology: SOMTopology
): tf.Tensor2D {
  const coords: number[][] = [];
  
  for (let row = 0; row < grid_height; row++) {
    for (let col = 0; col < grid_width; col++) {
      if (topology === 'rectangular') {
        coords.push([col, row]);
      } else {
        // Hexagonal coordinates with offset
        const x = col + (row % 2) * 0.5;
        const y = row * Math.sqrt(3) / 2;
        coords.push([x, y]);
      }
    }
  }
  
  return tf.tensor2d(coords);
}

/**
 * ----------------------------------------------------------------------------
 * Best Matching Unit (BMU) Calculation Functions
 * ----------------------------------------------------------------------------
 */

/**
 * Find the Best Matching Unit for a single sample.
 * 
 * @param sample Input sample tensor [n_features]
 * @param weights SOM weights tensor [grid_height, grid_width, n_features]
 * @returns BMU indices as [row, col]
 */
export function find_bmu(
  sample: tf.Tensor1D,
  weights: tf.Tensor3D
): tf.Tensor1D {
  return tf.tidy(() => {
    const [grid_height, grid_width, n_features] = weights.shape;
    
    // Reshape weights to 2D for efficient computation
    // [grid_height * grid_width, n_features]
    const weights_flat = weights.reshape([grid_height * grid_width, n_features]);
    
    // Compute squared distances to all neurons
    // ||sample - weight||^2 = ||sample||^2 + ||weight||^2 - 2 * sample . weight
    const sample_norm = sample.square().sum().expandDims(0);
    const weights_norm = weights_flat.square().sum(1, true);
    const dot_product = tf.mat_mul(
      sample.expandDims(0),
      weights_flat,
      false,
      true
    );
    
    const distances = sample_norm.add(weights_norm).sub(dot_product.mul(2));
    
    // Find minimum distance index
    const bmu_index = distances.argMin(1);
    const bmu_index_array = bmu_index.arraySync();
    const bmu_index_value: number = Array.isArray(bmu_index_array) 
      ? bmu_index_array[0] as number
      : bmu_index_array as number;
    
    // Convert flat index back to grid coordinates  
    const row_value = Math.floor(bmu_index_value / grid_width);
    const col_value = bmu_index_value % grid_width;
    
    return tf.tensor1d([row_value, col_value]);
  });
}

/**
 * Find BMUs for a batch of samples using optimized matrix operations.
 * 
 * @param samples Batch of input samples [n_samples, n_features]
 * @param weights SOM weights tensor [grid_height, grid_width, n_features]
 * @returns BMU indices tensor [n_samples, 2] with [row, col] for each sample
 */
export function find_bmu_batch(
  samples: tf.Tensor2D,
  weights: tf.Tensor3D
): tf.Tensor2D {
  return tf.tidy(() => {
    const [_n_samples, n_features] = samples.shape;
    const [grid_height, grid_width, _n_features_weight] = weights.shape;
    const total_neurons = grid_height * grid_width;
    
    // Reshape weights for batch computation
    // [total_neurons, n_features]
    const weights_flat = weights.reshape([total_neurons, n_features]);
    
    // Compute pairwise squared distances using broadcasting
    // Distance matrix will be [n_samples, total_neurons]
    
    // ||x - w||^2 = ||x||^2 + ||w||^2 - 2 * x . w
    const samples_norm = samples.square().sum(1, true); // [n_samples, 1]
    const weights_norm = weights_flat.square().sum(1, true).transpose(); // [1, total_neurons]
    
    // Dot product: [n_samples, n_features] x [n_features, total_neurons]
    const dot_product = tf.mat_mul(samples, weights_flat, false, true);
    
    // Compute distances
    const distances = samples_norm.add(weights_norm).sub(dot_product.mul(2));
    
    // Find BMU indices for each sample
    const bmu_indices = distances.argMin(1); // [n_samples]
    
    // Convert flat indices to grid coordinates
    const rows = bmu_indices.div(grid_width).floor();
    const cols = bmu_indices.mod(grid_width);
    
    return tf.stack([rows, cols], 1) as tf.Tensor2D;
  });
}

/**
 * Compute distances from samples to their BMUs.
 * Used for quantization error calculation.
 * 
 * @param samples Input samples [n_samples, n_features]
 * @param weights SOM weights [grid_height, grid_width, n_features]
 * @param bmus BMU indices [n_samples, 2]
 * @returns Distances tensor [n_samples]
 */
export function compute_bmu_distances(
  samples: tf.Tensor2D,
  weights: tf.Tensor3D,
  bmus: tf.Tensor2D
): tf.Tensor1D {
  // Get data outside tf.tidy to avoid disposal issues
  const [_n_samples, _n_features] = samples.shape;
  const [grid_height, grid_width, _n_features_weight] = weights.shape;
  const bmus_array = bmus.arraySync();
  
  return tf.tidy(() => {
    // Reshape weights for easier indexing
    const weights_flat = weights.reshape([grid_height * grid_width, _n_features]);
    
    // Convert BMU coordinates to flat indices
    const bmu_indices: number[] = [];
    
    for (let i = 0; i < _n_samples; i++) {
      const row = bmus_array[i][0];
      const col = bmus_array[i][1];
      bmu_indices.push(row * grid_width + col);
    }
    
    // Gather BMU weights using indices
    const bmu_indices_tensor = tf.tensor1d(bmu_indices, 'int32');
    const bmu_weights = tf.gather(weights_flat, bmu_indices_tensor);
    
    // Compute distances
    const diff = samples.sub(bmu_weights);
    const distances = diff.square().sum(1).sqrt();
    
    return distances as tf.Tensor1D;
  });
}

/**
 * ----------------------------------------------------------------------------
 * Neighborhood Functions for Weight Updates
 * ----------------------------------------------------------------------------
 */

/**
 * Gaussian neighborhood function.
 * Smooth exponential decay from BMU.
 * 
 * @param distance Distance from BMU in grid space
 * @param radius Current neighborhood radius
 * @returns Influence value [0, 1]
 */
export function gaussian_neighborhood(
  distance: tf.Tensor,
  radius: number
): tf.Tensor {
  return tf.tidy(() => {
    // h(d, σ) = exp(-d² / (2σ²))
    const sigma2 = 2 * radius * radius;
    return distance.square().div(sigma2).neg().exp();
  });
}

/**
 * Bubble neighborhood function.
 * Hard cutoff at radius boundary.
 * 
 * @param distance Distance from BMU in grid space
 * @param radius Current neighborhood radius
 * @returns Influence value (0 or 1)
 */
export function bubble_neighborhood(
  distance: tf.Tensor,
  radius: number
): tf.Tensor {
  return tf.tidy(() => {
    // h(d, σ) = 1 if d <= σ, else 0
    return distance.lessEqual(radius).cast('float32');
  });
}

/**
 * Mexican hat (Ricker wavelet) neighborhood function.
 * Provides lateral inhibition with negative values at medium distances.
 * 
 * @param distance Distance from BMU in grid space
 * @param radius Current neighborhood radius
 * @returns Influence value with positive center and negative surround
 */
export function mexican_hat_neighborhood(
  distance: tf.Tensor,
  radius: number
): tf.Tensor {
  return tf.tidy(() => {
    // Ricker wavelet formula:
    // h(d, σ) = A * (1 - (d/σ)²) * exp(-(d/σ)²/2)
    // where A is normalization constant (we use 2 for stronger effect)
    
    const sigma = radius;
    const dist_norm = distance.div(sigma);
    const dist_norm_squared = dist_norm.square();
    
    // Core Ricker wavelet
    const term1 = tf.scalar(1).sub(dist_norm_squared);
    const term2 = dist_norm_squared.div(2).neg().exp();
    const wavelet = term1.mul(term2);
    
    // Apply amplitude scaling for stronger lateral inhibition
    const amplitude = 2.0;
    return wavelet.mul(amplitude);
  });
}

/**
 * Compute neighborhood influence for all neurons given BMU positions.
 * 
 * @param bmus BMU positions [n_samples, 2]
 * @param grid_height Grid height
 * @param grid_width Grid width
 * @param radius Current neighborhood radius
 * @param neighborhood Neighborhood function type
 * @param topology Grid topology
 * @returns Influence matrix [n_samples, grid_height * grid_width]
 */
export function compute_neighborhood_influence(
  bmus: tf.Tensor2D,
  grid_height: number,
  grid_width: number,
  radius: number,
  neighborhood: SOMNeighborhood,
  topology: SOMTopology
): tf.Tensor2D {
  return tf.tidy(() => {
    const [_n_samples] = bmus.shape;
    const total_neurons = grid_height * grid_width;
    
    // Pre-compute grid distance matrix if not cached
    const grid_distances = create_grid_distance_matrix(grid_height, grid_width, topology);
    
    // Get BMU flat indices
    const bmus_data = bmus.bufferSync();
    const bmu_indices: number[] = [];
    for (let i = 0; i < _n_samples; i++) {
      const row = bmus_data.get(i, 0);
      const col = bmus_data.get(i, 1);
      bmu_indices.push(row * grid_width + col);
    }
    
    // Compute influence for each sample's BMU
    const influences: tf.Tensor[] = [];
    
    for (const bmu_idx of bmu_indices) {
      // Get distances from this BMU to all neurons
      const distances = grid_distances.slice([bmu_idx, 0], [1, total_neurons]).squeeze();
      
      // Apply neighborhood function
      let influence: tf.Tensor;
      switch (neighborhood) {
        case 'gaussian':
          influence = gaussian_neighborhood(distances, radius);
          break;
        case 'bubble':
          influence = bubble_neighborhood(distances, radius);
          break;
        case 'mexican_hat':
          influence = mexican_hat_neighborhood(distances, radius);
          break;
        default:
          throw new Error(`Unknown neighborhood function: ${neighborhood}`);
      }
      
      influences.push(influence.expandDims(0));
    }
    
    // Stack influences for all samples
    return tf.concat(influences, 0) as tf.Tensor2D;
  });
}

/**
 * Compute neighborhood influence matrix for batch update.
 * Optimized version that computes influence for all BMUs at once.
 * 
 * @param bmu_indices Flat BMU indices [n_samples]
 * @param grid_distance_matrix Pre-computed grid distances [total_neurons, total_neurons]
 * @param radius Current neighborhood radius
 * @param neighborhood Neighborhood function type
 * @returns Influence matrix [n_samples, total_neurons]
 */
export function compute_neighborhood_influence_batch(
  bmu_indices: tf.Tensor1D,
  grid_distance_matrix: tf.Tensor2D,
  radius: number,
  neighborhood: SOMNeighborhood
): tf.Tensor2D {
  return tf.tidy(() => {
    // Gather distances for all BMUs at once
    const bmu_distances = tf.gather(grid_distance_matrix, bmu_indices);
    
    // Apply neighborhood function to all distances
    switch (neighborhood) {
      case 'gaussian':
        return gaussian_neighborhood(bmu_distances, radius) as tf.Tensor2D;
      case 'bubble':
        return bubble_neighborhood(bmu_distances, radius) as tf.Tensor2D;
      case 'mexican_hat':
        return mexican_hat_neighborhood(bmu_distances, radius) as tf.Tensor2D;
      default:
        throw new Error(`Unknown neighborhood function: ${neighborhood}`);
    }
  });
}

/**
 * Create a cached neighborhood influence lookup table.
 * Pre-computes influence values for all possible neuron pairs.
 * 
 * @param grid_height Grid height
 * @param grid_width Grid width
 * @param radius Neighborhood radius
 * @param neighborhood Neighborhood function type
 * @param topology Grid topology
 * @returns Influence lookup table [total_neurons, total_neurons]
 */
export function create_neighborhood_lookup_table(
  grid_height: number,
  grid_width: number,
  radius: number,
  neighborhood: SOMNeighborhood,
  topology: SOMTopology
): tf.Tensor2D {
  return tf.tidy(() => {
    const grid_distances = create_grid_distance_matrix(grid_height, grid_width, topology);
    
    // Apply neighborhood function to entire distance matrix
    switch (neighborhood) {
      case 'gaussian':
        return gaussian_neighborhood(grid_distances, radius) as tf.Tensor2D;
      case 'bubble':
        return bubble_neighborhood(grid_distances, radius) as tf.Tensor2D;
      case 'mexican_hat':
        return mexican_hat_neighborhood(grid_distances, radius) as tf.Tensor2D;
      default:
        throw new Error(`Unknown neighborhood function: ${neighborhood}`);
    }
  });
}

/**
 * Validate neighborhood function parameters.
 * 
 * @param radius Neighborhood radius
 * @param grid_height Grid height
 * @param grid_width Grid width
 * @throws Error if parameters are invalid
 */
export function validate_neighborhood_params(
  radius: number,
  grid_height: number,
  grid_width: number
): void {
  if (radius <= 0) {
    throw new Error('Neighborhood radius must be positive');
  }
  
  const max_dimension = Math.max(grid_height, grid_width);
  if (radius > max_dimension) {
    console.warn(
      `Neighborhood radius (${radius}) is larger than grid dimensions. ` +
      `This may lead to uniform influence across the entire grid.`
    );
  }
}

/**
 * ----------------------------------------------------------------------------
 * Decay Strategies for Learning Rate and Radius
 * ----------------------------------------------------------------------------
 */

/**
 * Linear decay function.
 * Decreases linearly from initial to final value.
 * 
 * @param initial Initial value
 * @param final Final value (minimum)
 * @param epoch Current epoch
 * @param total_epochs Total number of epochs
 * @returns Decayed value
 */
export function linear_decay(
  initial: number,
  final: number,
  epoch: number,
  total_epochs: number
): number {
  if (total_epochs <= 1) return initial;
  const progress = epoch / (total_epochs - 1);
  return initial * (1 - progress) + final * progress;
}

/**
 * Exponential decay function.
 * Decreases exponentially with configurable decay rate.
 * 
 * @param initial Initial value
 * @param final Final value (minimum)
 * @param epoch Current epoch
 * @param total_epochs Total number of epochs
 * @param decay_rate Decay rate (default: 5)
 * @returns Decayed value
 */
export function exponential_decay(
  initial: number,
  final: number,
  epoch: number,
  total_epochs: number,
  decay_rate: number = 5
): number {
  if (total_epochs <= 1) return initial;
  const time_constant = total_epochs / decay_rate;
  const decay_factor = Math.exp(-epoch / time_constant);
  return final + (initial - final) * decay_factor;
}

/**
 * Inverse time decay function.
 * Decreases using inverse time formula.
 * 
 * @param initial Initial value
 * @param final Final value (minimum)
 * @param epoch Current epoch
 * @param total_epochs Total number of epochs
 * @param decay_rate Decay rate (default: 1)
 * @returns Decayed value
 */
export function inverse_time_decay(
  initial: number,
  final: number,
  epoch: number,
  total_epochs: number,
  decay_rate: number = 1
): number {
  if (total_epochs <= 1) return initial;
  const decay_factor = 1 / (1 + decay_rate * epoch / total_epochs);
  return final + (initial - final) * decay_factor;
}

/**
 * Create a decay scheduler for learning rate or radius.
 * 
 * @param initial Initial value or custom decay function
 * @param strategy Decay strategy name
 * @param total_epochs Total number of epochs
 * @param final Final value (minimum)
 * @returns Decay function
 */
export function create_decay_scheduler(
  initial: number | DecayFunction,
  strategy: 'linear' | 'exponential' | 'inverse' = 'exponential',
  total_epochs: number,
  final?: number
): DecayFunction {
  // If already a function, return it
  if (typeof initial === 'function') {
    return initial;
  }
  
  // Set default final value
  const final_value = final ?? initial * 0.01;
  
  // Return appropriate decay function
  switch (strategy) {
    case 'linear':
      return (epoch: number) => linear_decay(initial, final_value, epoch, total_epochs);
    case 'exponential':
      return (epoch: number) => exponential_decay(initial, final_value, epoch, total_epochs);
    case 'inverse':
      return (epoch: number) => inverse_time_decay(initial, final_value, epoch, total_epochs);
    default:
      throw new Error(`Unknown decay strategy: ${strategy}`);
  }
}

/**
 * Decay scheduler with TensorFlow.js tensors for GPU computation.
 * Used when decay needs to be computed on GPU for performance.
 * 
 * @param initial Initial value tensor
 * @param final Final value tensor
 * @param epoch Current epoch tensor
 * @param total_epochs Total epochs tensor
 * @param strategy Decay strategy
 * @returns Decayed value tensor
 */
export function decay_tensor(
  initial: tf.Tensor | number,
  final: tf.Tensor | number,
  epoch: tf.Tensor | number,
  total_epochs: tf.Tensor | number,
  strategy: 'linear' | 'exponential' | 'inverse' = 'exponential'
): tf.Tensor {
  return tf.tidy(() => {
    // Convert to tensors if needed
    const initial_t = typeof initial === 'number' ? tf.scalar(initial) : initial;
    const final_t = typeof final === 'number' ? tf.scalar(final) : final;
    const epoch_t = typeof epoch === 'number' ? tf.scalar(epoch) : epoch;
    const total_epochs_t = typeof total_epochs === 'number' ? tf.scalar(total_epochs) : total_epochs;
    
    switch (strategy) {
      case 'linear': {
        // linear: initial * (1 - t) + final * t
        const progress = epoch_t.div(total_epochs_t.sub(1));
        const term1 = initial_t.mul(tf.scalar(1).sub(progress));
        const term2 = final_t.mul(progress);
        return term1.add(term2);
      }
      
      case 'exponential': {
        // exponential: final + (initial - final) * exp(-epoch / time_constant)
        const time_constant = total_epochs_t.div(5); // decay rate = 5
        const decay_factor = epoch_t.div(time_constant).neg().exp();
        return final_t.add(initial_t.sub(final_t).mul(decay_factor));
      }
      
      case 'inverse': {
        // inverse: final + (initial - final) / (1 + epoch / total_epochs)
        const decay_factor = tf.scalar(1).div(
          tf.scalar(1).add(epoch_t.div(total_epochs_t))
        );
        return final_t.add(initial_t.sub(final_t).mul(decay_factor));
      }
      
      default:
        throw new Error(`Unknown decay strategy: ${strategy}`);
    }
  });
}

/**
 * Track decay schedule across epochs.
 * Maintains history of decayed values for visualization.
 */
export class DecayTracker {
  private history: number[] = [];
  private decay_fn: DecayFunction;
  private current_epoch: number = 0;
  
  constructor(
    initial: number | DecayFunction,
    strategy: 'linear' | 'exponential' | 'inverse' = 'exponential',
    total_epochs: number,
    final?: number
  ) {
    this.decay_fn = create_decay_scheduler(initial, strategy, total_epochs, final);
  }
  
  /**
   * Get current value and advance epoch.
   */
  next(total_epochs: number): number {
    const value = this.decay_fn(this.current_epoch, total_epochs);
    this.history.push(value);
    this.current_epoch++;
    return value;
  }
  
  /**
   * Get current value without advancing.
   */
  current(total_epochs: number): number {
    return this.decay_fn(this.current_epoch, total_epochs);
  }
  
  /**
   * Reset tracker to initial state.
   */
  reset(): void {
    this.current_epoch = 0;
    this.history = [];
  }
  
  /**
   * Get decay history.
   */
  get_history(): number[] {
    return [...this.history];
  }
  
  /**
   * Get current epoch.
   */
  get_epoch(): number {
    return this.current_epoch;
  }
}

/**
 * Calculate adaptive radius based on grid size.
 * 
 * @param grid_height Grid height
 * @param grid_width Grid width
 * @param epoch Current epoch
 * @param total_epochs Total epochs
 * @returns Adaptive radius value
 */
export function adaptive_radius(
  grid_height: number,
  grid_width: number,
  epoch: number,
  total_epochs: number
): number {
  const initial_radius = Math.max(grid_height, grid_width) / 2;
  const final_radius = 1;
  return exponential_decay(initial_radius, final_radius, epoch, total_epochs);
}

/**
 * Calculate adaptive learning rate.
 * 
 * @param initial Initial learning rate
 * @param epoch Current epoch
 * @param total_epochs Total epochs
 * @returns Adaptive learning rate
 */
export function adaptive_learning_rate(
  initial: number,
  epoch: number,
  total_epochs: number
): number {
  const final = initial * 0.01;
  return exponential_decay(initial, final, epoch, total_epochs);
}