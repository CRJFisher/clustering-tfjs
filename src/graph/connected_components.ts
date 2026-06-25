import * as tf from '../backend/adapter';
import { SparseMatrix } from './sparse';

export function detect_connected_components(
  affinity: tf.Tensor2D,
  tolerance: number = 1e-2,
): {
  num_components: number;
  is_fully_connected: boolean;
  component_labels: Int32Array;
} {
  const n = affinity.shape[0];
  const component_labels = new Int32Array(n).fill(-1);

  const affinity_data = affinity.arraySync();
  let current_component = 0;

  for (let start_node = 0; start_node < n; start_node++) {
    if (component_labels[start_node] !== -1) continue;

    const queue: number[] = [start_node];
    component_labels[start_node] = current_component;

    while (queue.length > 0) {
      const node = queue.shift()!;

      for (let neighbor = 0; neighbor < n; neighbor++) {
        if (
          neighbor !== node &&
          component_labels[neighbor] === -1 &&
          affinity_data[node][neighbor] > tolerance
        ) {
          component_labels[neighbor] = current_component;
          queue.push(neighbor);
        }
      }
    }

    current_component++;
  }

  const num_components = current_component;

  return {
    num_components,
    is_fully_connected: num_components === 1,
    component_labels,
  };
}

/** @throws If the affinity matrix is not square. */
export function detect_sparse_connected_components(
  affinity: SparseMatrix,
  tolerance: number = 1e-2,
): {
  num_components: number;
  is_fully_connected: boolean;
  component_labels: Int32Array;
} {
  if (affinity.rows !== affinity.cols) {
    throw new Error('Affinity matrix must be square (n × n).');
  }

  const n = affinity.rows;
  const component_labels = new Int32Array(n).fill(-1);
  let current_component = 0;

  for (let start_node = 0; start_node < n; start_node++) {
    if (component_labels[start_node] !== -1) continue;

    const queue: number[] = [start_node];
    component_labels[start_node] = current_component;

    // Index-based iteration avoids the O(n) cost of array.shift() on each dequeue.
    for (let queue_idx = 0; queue_idx < queue.length; queue_idx++) {
      const node = queue[queue_idx];
      for (
        let ptr = affinity.indptr[node];
        ptr < affinity.indptr[node + 1];
        ptr++
      ) {
        const neighbor = affinity.indices[ptr];
        if (
          neighbor !== node &&
          component_labels[neighbor] === -1 &&
          affinity.data[ptr] > tolerance
        ) {
          component_labels[neighbor] = current_component;
          queue.push(neighbor);
        }
      }
    }

    current_component++;
  }

  return {
    num_components: current_component,
    is_fully_connected: current_component === 1,
    component_labels,
  };
}

