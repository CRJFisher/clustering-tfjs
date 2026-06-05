import tfDefault from '../backend/adapter';
import * as tf from '../backend/adapter';
import type { SOM } from '../clustering/som';

/**
 * SOM Visualization Utilities
 * 
 * Provides functions for visualizing and analyzing trained SOM models.
 */

/**
 * Calculate component planes for each feature.
 * Shows how each input feature is distributed across the SOM grid.
 * 
 * @param som Trained SOM instance
 * @returns Component planes tensor [n_features, grid_height, grid_width]
 */
export function get_component_planes(som: SOM): tf.Tensor3D {
  const weights = som.get_weights();

  return tf.tidy(() => {
    const weights_tensor = tf.tensor3d(weights);
    // Transpose from [height, width, features] to [features, height, width]
    return weights_tensor.transpose([2, 0, 1]);
  });
}

/**
 * Calculate hit map showing sample distribution across neurons.
 * 
 * @param som Trained SOM instance
 * @param X Input data used for training
 * @returns Hit map tensor [grid_height, grid_width] with counts
 */
export async function get_hit_map(
  som: SOM,
  X: tf.Tensor2D
): Promise<tf.Tensor2D> {
  const { grid_height, grid_width } = som.params;
  
  // Get BMUs for all samples
  const labels = await som.predict(X);
  
  // Count hits for each neuron
  const hit_map = tf.buffer([grid_height, grid_width]);
  
  if (Array.isArray(labels)) {
    for (const label of labels) {
      const row = Math.floor(label / grid_width);
      const col = label % grid_width;
      hit_map.set(hit_map.get(row, col) + 1, row, col);
    }
  }
  
  return hit_map.toTensor() as tf.Tensor2D;
}

/**
 * Calculate activation map for a specific input sample.
 * Shows how strongly each neuron responds to the input.
 * 
 * @param som Trained SOM instance
 * @param sample Input sample tensor [n_features]
 * @returns Activation map [grid_height, grid_width]
 */
export function get_activation_map(
  som: SOM,
  sample: tf.Tensor1D
): tf.Tensor2D {
  const weights = som.get_weights();
  const { grid_height, grid_width } = som.params;
  const n_features = weights[0][0].length;

  return tf.tidy(() => {
    const weights_tensor = tf.tensor3d(weights);

    // Reshape weights for computation
    const weights_flat = weights_tensor.reshape([grid_height * grid_width, n_features]);

    // Compute distances from sample to all neurons
    const diff = weights_flat.sub(sample.expandDims(0));
    const distances = diff.square().sum(1).sqrt();

    // Convert distances to activations (inverse relationship)
    const max_dist = distances.max();
    const activations = max_dist.sub(distances).div(max_dist);

    // Reshape to grid
    return activations.reshape([grid_height, grid_width]) as tf.Tensor2D;
  });
}

/**
 * Track BMU trajectory for a sequence of samples.
 * Useful for analyzing temporal patterns.
 * 
 * @param som Trained SOM instance
 * @param sequence Sequence of samples [n_samples, n_features]
 * @returns BMU positions [n_samples, 2] with grid coordinates
 */
export async function track_bmu_trajectory(
  som: SOM,
  sequence: tf.Tensor2D
): Promise<number[][]> {
  const { grid_width } = som.params;
  const labels = await som.predict(sequence);
  
  const trajectory: number[][] = [];
  if (Array.isArray(labels)) {
    for (const label of labels) {
      const row = Math.floor(label / grid_width);
      const col = label % grid_width;
      trajectory.push([row, col]);
    }
  }
  
  return trajectory;
}

/**
 * Calculate quantization quality map.
 * Shows average quantization error for each neuron.
 * 
 * @param som Trained SOM instance
 * @param X Input data
 * @returns Quality map [grid_height, grid_width]
 */
export async function get_quantization_quality_map(
  som: SOM,
  X: tf.Tensor2D
): Promise<tf.Tensor2D> {
  const { grid_height, grid_width } = som.params;
  const weights_array = som.get_weights();

  // Get labels and compute distances
  const labels = await som.predict(X);

  // Initialize quality map
  const quality_map = tf.buffer([grid_height, grid_width]);
  const counts = tf.buffer([grid_height, grid_width]);

  // Compute average distance for each neuron
  const x_array = await X.array();
  
  if (Array.isArray(labels)) {
    for (let i = 0; i < labels.length; i++) {
      const label = labels[i];
      const row = Math.floor(label / grid_width);
      const col = label % grid_width;
      
      // Compute distance
      const sample = x_array[i];
      const weight = weights_array[row][col];
      const distance = Math.sqrt(
        sample.reduce((sum, val, idx) => 
          sum + Math.pow(val - weight[idx], 2), 0
        )
      );
      
      quality_map.set(
        quality_map.get(row, col) + distance,
        row,
        col
      );
      counts.set(counts.get(row, col) + 1, row, col);
    }
  }
  
  // Average the distances
  for (let i = 0; i < grid_height; i++) {
    for (let j = 0; j < grid_width; j++) {
      const count = counts.get(i, j);
      if (count > 0) {
        quality_map.set(quality_map.get(i, j) / count, i, j);
      }
    }
  }
  
  return quality_map.toTensor() as tf.Tensor2D;
}

/**
 * Generate neighbor distance matrix for topology preservation analysis.
 * 
 * @param som Trained SOM instance
 * @returns Neighbor distances [grid_height, grid_width]
 */
export function get_neighbor_distance_matrix(som: SOM): tf.Tensor2D {
  const weights_array = som.get_weights();
  const { grid_height, grid_width, topology } = som.params;

  return tf.tidy(() => {
    const distance_map = tf.buffer([grid_height, grid_width]);
    
    for (let i = 0; i < grid_height; i++) {
      for (let j = 0; j < grid_width; j++) {
        const current_weight = weights_array[i][j];
        let total_distance = 0;
        let neighbor_count = 0;
        
        // Define neighbors based on topology
        const neighbors: [number, number][] = [];
        if (topology === 'rectangular') {
          // 8-connected neighbors (consistent with get_u_matrix and are_neighbors)
          const deltas = [
            [-1, -1], [-1, 0], [-1, 1],
            [0, -1],           [0, 1],
            [1, -1],  [1, 0],  [1, 1]
          ];
          for (const [di, dj] of deltas) {
            const ni = i + di;
            const nj = j + dj;
            if (ni >= 0 && ni < grid_height && nj >= 0 && nj < grid_width) {
              neighbors.push([ni, nj]);
            }
          }
        } else {
          // Hexagonal neighbors (6-connected)
          const even_row = i % 2 === 0;
          if (i > 0) {
            neighbors.push([i - 1, j]);
            if (even_row && j > 0) neighbors.push([i - 1, j - 1]);
            if (!even_row && j < grid_width - 1) neighbors.push([i - 1, j + 1]);
          }
          if (i < grid_height - 1) {
            neighbors.push([i + 1, j]);
            if (even_row && j > 0) neighbors.push([i + 1, j - 1]);
            if (!even_row && j < grid_width - 1) neighbors.push([i + 1, j + 1]);
          }
          if (j > 0) neighbors.push([i, j - 1]);
          if (j < grid_width - 1) neighbors.push([i, j + 1]);
        }
        
        // Calculate average distance to neighbors
        for (const [ni, nj] of neighbors) {
          const neighbor_weight = weights_array[ni][nj];
          const distance = Math.sqrt(
            current_weight.reduce((sum, val, idx) => 
              sum + Math.pow(val - neighbor_weight[idx], 2), 0
            )
          );
          total_distance += distance;
          neighbor_count++;
        }
        
        distance_map.set(
          neighbor_count > 0 ? total_distance / neighbor_count : 0,
          i,
          j
        );
      }
    }
    
    return distance_map.toTensor() as tf.Tensor2D;
  });
}

/**
 * Export SOM data for external visualization tools.
 * 
 * @param som Trained SOM instance
 * @param format Export format ('json', 'csv')
 * @returns Formatted string for export
 */
export async function export_for_visualization(
  som: SOM,
  format: 'json' | 'csv' = 'json'
): Promise<string> {
  const weights_array = som.get_weights();
  const u_matrix = som.get_u_matrix();
  const { grid_height, grid_width } = som.params;

  try {
    const u_matrix_array = await u_matrix.array();

    if (format === 'json') {
      return JSON.stringify({
        grid_height,
        grid_width,
        weights: weights_array,
        u_matrix: u_matrix_array,
        params: som.params,
      }, null, 2);
    } else {
      // CSV format
      let csv = 'row,col,u_value';
      for (let i = 0; i < weights_array[0][0].length; i++) {
        csv += `,feature_${i}`;
      }
      csv += '\n';

      for (let i = 0; i < grid_height; i++) {
        for (let j = 0; j < grid_width; j++) {
          csv += `${i},${j},${u_matrix_array[i][j]}`;
          for (const feature of weights_array[i][j]) {
            csv += `,${feature}`;
          }
          csv += '\n';
        }
      }

      return csv;
    }
  } finally {
    u_matrix.dispose();
  }
}

/**
 * Calculate density map showing data concentration.
 * 
 * @param som Trained SOM instance
 * @param X Input data
 * @param sigma Gaussian kernel width for smoothing
 * @returns Density map [grid_height, grid_width]
 */
export async function get_density_map(
  som: SOM,
  X: tf.Tensor2D,
  sigma: number = 1.0
): Promise<tf.Tensor2D> {
  const hit_map = await get_hit_map(som, X);

  // No smoothing needed for sigma <= 0
  if (sigma <= 0) {
    return hit_map;
  }

  try {
    return tf.tidy(() => {
      // Create Gaussian kernel
      const kernel_size = Math.ceil(sigma * 3) * 2 + 1;
      const kernel = tf.buffer([kernel_size, kernel_size]);
      const center = Math.floor(kernel_size / 2);

      let sum = 0;
      for (let i = 0; i < kernel_size; i++) {
        for (let j = 0; j < kernel_size; j++) {
          const distance = Math.sqrt(
            Math.pow(i - center, 2) + Math.pow(j - center, 2)
          );
          const value = Math.exp(-distance * distance / (2 * sigma * sigma));
          kernel.set(value, i, j);
          sum += value;
        }
      }

      // Normalize kernel
      for (let i = 0; i < kernel_size; i++) {
        for (let j = 0; j < kernel_size; j++) {
          kernel.set(kernel.get(i, j) / sum, i, j);
        }
      }

      // Reshape hit_map [H, W] -> [1, H, W, 1] for conv2d
      const input = hit_map.expandDims(0).expandDims(-1) as tf.Tensor4D;

      // Reshape kernel [K, K] -> [K, K, 1, 1] for conv2d filter
      const kernel_tensor = kernel.toTensor();
      const filter = kernel_tensor.expandDims(-1).expandDims(-1) as tf.Tensor4D;

      // Apply 2D convolution with 'same' padding to preserve dimensions
      const convolved = tfDefault.conv2d(input, filter, 1, 'same');

      // Squeeze back from [1, H, W, 1] to [H, W]
      return convolved.squeeze([0, 3]) as tf.Tensor2D;
    });
  } finally {
    hit_map.dispose();
  }
}