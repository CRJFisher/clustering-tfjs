import * as tf from '../backend/adapter';

/**
 * Detects the number of connected components in a graph based on its affinity matrix.
 *
 * A graph is considered fully connected if there's only 1 component.
 * Multiple components are detected by counting near-zero eigenvalues of the Laplacian.
 *
 * @param affinity - Affinity/adjacency matrix (n x n)
 * @param tolerance - Tolerance for detecting zero eigenvalues (default: 1e-2)
 * @returns Object containing:
 *   - num_components: Number of connected components
 *   - is_fully_connected: Whether the graph has only 1 component
 *   - component_labels: Array indicating which component each node belongs to
 */
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

  // Get affinity data for graph traversal
  const affinity_data = affinity.arraySync();

  // BFS to find connected components
  let current_component = 0;

  for (let start_node = 0; start_node < n; start_node++) {
    if (component_labels[start_node] !== -1) continue; // Already assigned

    // BFS from this node
    const queue: number[] = [start_node];
    component_labels[start_node] = current_component;

    while (queue.length > 0) {
      const node = queue.shift()!;

      // Check all neighbors
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

/**
 * Issues a warning if the graph is not fully connected, similar to sklearn.
 *
 * @param affinity - Affinity matrix to check
 * @param tolerance - Tolerance for detecting zero eigenvalues
 * @returns true if graph is fully connected, false otherwise
 */
export function check_graph_connectivity(
  affinity: tf.Tensor2D,
  tolerance: number = 1e-2,
): boolean {
  const { is_fully_connected } = detect_connected_components(affinity, tolerance);

  if (!is_fully_connected) {
    console.warn(
      'Graph is not fully connected, spectral embedding may not work as expected.',
    );
  }

  return is_fully_connected;
}
