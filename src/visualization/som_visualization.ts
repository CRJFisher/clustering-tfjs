import tfDefault from '../backend/adapter';
import * as tf from '../backend/adapter';
import type { SOM } from '../clustering/som';

export function get_component_planes(som: SOM): tf.Tensor3D {
  const weights = som.get_weights();

  return tf.tidy(() => {
    const weights_tensor = tf.tensor3d(weights);
    return weights_tensor.transpose([2, 0, 1]);
  });
}

export async function get_hit_map(
  som: SOM,
  X: tf.Tensor2D
): Promise<tf.Tensor2D> {
  const { grid_height, grid_width } = som.params;

  const labels = await som.predict(X);

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

export function get_activation_map(
  som: SOM,
  sample: tf.Tensor1D
): tf.Tensor2D {
  const weights = som.get_weights();
  const { grid_height, grid_width } = som.params;
  const n_features = weights[0][0].length;

  return tf.tidy(() => {
    const weights_tensor = tf.tensor3d(weights);
    const weights_flat = weights_tensor.reshape([grid_height * grid_width, n_features]);

    const diff = weights_flat.sub(sample.expandDims(0));
    const distances = diff.square().sum(1).sqrt();

    // Activation is the distance inverted against the farthest neuron, so the
    // BMU (distance 0) maps to 1 and the most distant neuron maps to 0.
    const max_dist = distances.max();
    const activations = max_dist.sub(distances).div(max_dist);

    return activations.reshape([grid_height, grid_width]) as tf.Tensor2D;
  });
}

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

export async function get_quantization_quality_map(
  som: SOM,
  X: tf.Tensor2D
): Promise<tf.Tensor2D> {
  const { grid_height, grid_width } = som.params;
  const weights_array = som.get_weights();

  const labels = await som.predict(X);

  const quality_map = tf.buffer([grid_height, grid_width]);
  const counts = tf.buffer([grid_height, grid_width]);

  const x_array = await X.array();

  if (Array.isArray(labels)) {
    for (let i = 0; i < labels.length; i++) {
      const label = labels[i];
      const row = Math.floor(label / grid_width);
      const col = label % grid_width;

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

        const neighbors: [number, number][] = [];
        if (topology === 'rectangular') {
          // 8-connected neighbours, matching get_u_matrix and are_neighbors.
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
          // Hexagonal topology: 6-connected, with the diagonal pair shifting
          // by parity of the row.
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

export async function get_density_map(
  som: SOM,
  X: tf.Tensor2D,
  sigma: number = 1.0
): Promise<tf.Tensor2D> {
  const hit_map = await get_hit_map(som, X);

  if (sigma <= 0) {
    return hit_map;
  }

  try {
    return tf.tidy(() => {
      // Gaussian kernel spanning ±3σ, the radius beyond which weights are negligible.
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

      for (let i = 0; i < kernel_size; i++) {
        for (let j = 0; j < kernel_size; j++) {
          kernel.set(kernel.get(i, j) / sum, i, j);
        }
      }

      // conv2d needs NHWC input and HWIO filter; both spatial maps carry a
      // single channel here.
      const input = hit_map.expandDims(0).expandDims(-1) as tf.Tensor4D;
      const kernel_tensor = kernel.toTensor();
      const filter = kernel_tensor.expandDims(-1).expandDims(-1) as tf.Tensor4D;

      // tfDefault: conv2d is reached through the default proxy because it is
      // not one of the adapter's named exports.
      const convolved = tfDefault.conv2d(input, filter, 1, 'same');

      return convolved.squeeze([0, 3]) as tf.Tensor2D;
    });
  } finally {
    hit_map.dispose();
  }
}