import * as tf from '../backend/adapter';

/**
 * Creates component indicator features for disconnected graphs.
 *
 * For a graph with k connected components, this creates indicator features
 * where each feature has a constant value for all nodes in that component.
 * This mimics the behavior of sklearn's shift-invert eigenvectors.
 *
 * @param component_labels - Array indicating which component each node belongs to
 * @param num_components - Total number of components detected
 * @param max_indicators - Maximum number of indicator vectors to create (usually n_clusters)
 * @returns Component indicator matrix (n_samples x min(num_components, max_indicators))
 */
export function create_component_indicators(
  component_labels: Int32Array,
  num_components: number,
  max_indicators: number,
): tf.Tensor2D {
  return tf.tidy(() => {
    const n = component_labels.length;
    const num_indicators = Math.min(num_components, max_indicators);

    // Count nodes per component for normalization
    const component_sizes = new Array(num_components).fill(0);
    for (let i = 0; i < n; i++) {
      component_sizes[component_labels[i]]++;
    }

    // Create indicator matrix
    const indicators = new Float32Array(n * num_indicators);

    // Fill indicators with normalized values
    // Using 1/sqrt(component_size) normalization to match eigenvector normalization
    for (let i = 0; i < n; i++) {
      const comp = component_labels[i];
      if (comp < num_indicators) {
        indicators[i * num_indicators + comp] =
          1.0 / Math.sqrt(component_sizes[comp]);
      }
    }

    return tf.tensor2d(indicators, [n, num_indicators], 'float32');
  });
}
