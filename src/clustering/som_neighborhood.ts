import * as tf from '../backend/adapter';
import type { SOMTopology, SOMInitialization, SOMNeighborhood, DecayFunction } from './types';
import { power_iteration_eig } from '../decomposition/pca';

export function grid_to_index(row: number, col: number, grid_width: number): number {
  return row * grid_width + col;
}

export function index_to_grid(index: number, grid_width: number): [number, number] {
  const row = Math.floor(index / grid_width);
  const col = index % grid_width;
  return [row, col];
}

export function grid_distance(
  pos1: [number, number],
  pos2: [number, number],
  topology: SOMTopology
): number {
  const [r1, c1] = pos1;
  const [r2, c2] = pos2;
  
  if (topology === 'rectangular') {
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
        const x_min = X.min(0);
        const x_max = X.max(0);
        const range = x_max.sub(x_min);

        const random_weights = tf.random_uniform(
          [grid_height, grid_width, n_features],
          0,
          1,
          undefined,
          random_seed
        );
        
        return random_weights
          .mul(range.reshape([1, 1, n_features]))
          .add(x_min.reshape([1, 1, n_features]));
      }
      
      case 'linear': {
        // Following MiniSom's pca_weights_init approach
        const [n_samples_lin, n_features_lin] = X.shape;

        if (n_samples_lin < 2 || n_features_lin < 2) {
          return initialize_weights(X, grid_height, grid_width, 'random', random_seed);
        }

        const mean = X.mean(0, true);
        const centered = X.sub(mean);

        const cov = tf.mat_mul(centered, centered, true, false).div(n_samples_lin - 1);

        const n_comps = Math.min(2, n_features_lin);
        const components = compute_principal_components(cov as tf.Tensor2D, n_comps, random_seed);

        const projections = tf.mat_mul(centered, components, false, true);
        const proj_var = projections.square().sum(0).div(n_samples_lin - 1); // unbiased (centered data has mean 0)
        const proj_std = proj_var.sqrt();
        const proj_std_data = proj_std.dataSync();

        const components_data = components.arraySync() as number[][];
        const scaled_pc1 = components_data[0].map(v => v * proj_std_data[0]);
        const scaled_pc2 = n_comps > 1
          ? components_data[1].map(v => v * proj_std_data[1])
          : new Array(n_features_lin).fill(0);

        const mean_data = mean.squeeze().dataSync();

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
        const [_n_samples, n_features] = X.shape;
        
        if (_n_samples < 2) {
          // Fall back to random initialization if not enough samples
          return initialize_weights(X, grid_height, grid_width, 'random', random_seed);
        }

        const mean = X.mean(0, true);
        const centered = X.sub(mean);

        const cov = tf.mat_mul(centered, centered, true, false).div(_n_samples - 1);

        const n_comps = Math.min(2, n_features);
        const components = compute_principal_components(cov as tf.Tensor2D, n_comps, random_seed);

        const projections = tf.mat_mul(centered, components, false, true);
        const pc_min = projections.min(0);
        const pc_max = projections.max(0);
        const pc_range = pc_max.sub(pc_min);

        const weights: number[][][] = [];

        for (let i = 0; i < grid_height; i++) {
          const row_weights: number[][] = [];
          for (let j = 0; j < grid_width; j++) {
            const alpha = grid_height > 1 ? i / (grid_height - 1) : 0.5;
            const beta = grid_width > 1 ? j / (grid_width - 1) : 0.5;

            const pc_coords = tf.zeros([n_comps]);
            const pc_buffer = pc_coords.bufferSync();
            pc_buffer.set(pc_min.dataSync()[0] + pc_range.dataSync()[0] * alpha, 0);
            if (n_comps > 1) {
              pc_buffer.set(pc_min.dataSync()[1] + pc_range.dataSync()[1] * beta, 1);
            }

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
        const x = col + (row % 2) * 0.5;
        const y = row * Math.sqrt(3) / 2;
        coords.push([x, y]);
      }
    }
  }
  
  return tf.tensor2d(coords);
}

export function find_bmu(
  sample: tf.Tensor1D,
  weights: tf.Tensor3D
): tf.Tensor1D {
  return tf.tidy(() => {
    const [grid_height, grid_width, n_features] = weights.shape;

    const weights_flat = weights.reshape([grid_height * grid_width, n_features]);

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

    const bmu_index = distances.argMin(1);
    const bmu_index_array = bmu_index.arraySync();
    const bmu_index_value: number = Array.isArray(bmu_index_array) 
      ? bmu_index_array[0] as number
      : bmu_index_array as number;
    
    const row_value = Math.floor(bmu_index_value / grid_width);
    const col_value = bmu_index_value % grid_width;
    
    return tf.tensor1d([row_value, col_value]);
  });
}

export function find_bmu_batch(
  samples: tf.Tensor2D,
  weights: tf.Tensor3D
): tf.Tensor2D {
  return tf.tidy(() => {
    const [_n_samples, n_features] = samples.shape;
    const [grid_height, grid_width, _n_features_weight] = weights.shape;
    const total_neurons = grid_height * grid_width;

    const weights_flat = weights.reshape([total_neurons, n_features]);

    // ||x - w||^2 = ||x||^2 + ||w||^2 - 2 * x . w
    const samples_norm = samples.square().sum(1, true);
    const weights_norm = weights_flat.square().sum(1, true).transpose();

    const dot_product = tf.mat_mul(samples, weights_flat, false, true);

    const distances = samples_norm.add(weights_norm).sub(dot_product.mul(2));

    const bmu_indices = distances.argMin(1);

    const rows = bmu_indices.div(grid_width).floor();
    const cols = bmu_indices.mod(grid_width);
    
    return tf.stack([rows, cols], 1) as tf.Tensor2D;
  });
}

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
    const weights_flat = weights.reshape([grid_height * grid_width, _n_features]);

    const bmu_indices: number[] = [];
    
    for (let i = 0; i < _n_samples; i++) {
      const row = bmus_array[i][0];
      const col = bmus_array[i][1];
      bmu_indices.push(row * grid_width + col);
    }
    
    const bmu_indices_tensor = tf.tensor1d(bmu_indices, 'int32');
    const bmu_weights = tf.gather(weights_flat, bmu_indices_tensor);

    const diff = samples.sub(bmu_weights);
    const distances = diff.square().sum(1).sqrt();
    
    return distances as tf.Tensor1D;
  });
}

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

export function bubble_neighborhood(
  distance: tf.Tensor,
  radius: number
): tf.Tensor {
  return tf.tidy(() => {
    // h(d, σ) = 1 if d <= σ, else 0
    return distance.lessEqual(radius).cast('float32');
  });
}

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
    
    const term1 = tf.scalar(1).sub(dist_norm_squared);
    const term2 = dist_norm_squared.div(2).neg().exp();
    const wavelet = term1.mul(term2);
    
    // Apply amplitude scaling for stronger lateral inhibition
    const amplitude = 2.0;
    return wavelet.mul(amplitude);
  });
}

export function compute_neighborhood_influence_batch(
  bmu_indices: tf.Tensor1D,
  grid_distance_matrix: tf.Tensor2D,
  radius: number,
  neighborhood: SOMNeighborhood
): tf.Tensor2D {
  return tf.tidy(() => {
    const bmu_distances = tf.gather(grid_distance_matrix, bmu_indices);

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

export function create_decay_scheduler(
  initial: number | DecayFunction,
  strategy: 'linear' | 'exponential' | 'inverse' = 'exponential',
  total_epochs: number,
  final?: number
): DecayFunction {
  if (typeof initial === 'function') {
    return initial;
  }

  const final_value = final ?? initial * 0.01;

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

