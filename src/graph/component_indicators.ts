import * as tf from '../backend/adapter';

export function create_component_indicators(
  component_labels: Int32Array,
  num_components: number,
  max_indicators: number,
): tf.Tensor2D {
  return tf.tidy(() => {
    const n = component_labels.length;
    const num_indicators = Math.min(num_components, max_indicators);

    const component_sizes = new Array(num_components).fill(0);
    for (let i = 0; i < n; i++) {
      component_sizes[component_labels[i]]++;
    }

    const indicators = new Float32Array(n * num_indicators);

    // 1/sqrt(size) matches sklearn's shift-invert eigenvector normalization convention.
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
