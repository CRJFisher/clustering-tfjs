/**
 * Node.js-specific TensorFlow.js loader
 * 
 * This module attempts to load the best available TensorFlow.js backend
 * for Node.js environments, with automatic fallback.
 */

export async function load_tensor_flow() {
  // Try GPU backend first
  try {
    // Use require.resolve to check if module exists before importing
    require.resolve('@tensorflow/tfjs-node-gpu');
    const tf_gpu = await import('@tensorflow/tfjs-node-gpu' as string);
    console.log('Using TensorFlow.js GPU backend');
    return tf_gpu as typeof import('@tensorflow/tfjs');
  } catch {
    // GPU backend not available, try CPU backend
    try {
      require.resolve('@tensorflow/tfjs-node');
      const tf_node = await import('@tensorflow/tfjs-node');
      console.log('Using TensorFlow.js Node.js CPU backend');
      return tf_node as typeof import('@tensorflow/tfjs');
    } catch {
      // Neither Node backend available, fall back to pure JS
      try {
        require.resolve('@tensorflow/tfjs');
        const tf_js = await import('@tensorflow/tfjs');
        console.log('Using TensorFlow.js pure JavaScript backend (slower performance)');
        return tf_js as typeof import('@tensorflow/tfjs');
      } catch {
        throw new Error(
          'No TensorFlow.js backend available. Please install one of:\n' +
          '- @tensorflow/tfjs-node-gpu (for GPU acceleration)\n' +
          '- @tensorflow/tfjs-node (for CPU acceleration)\n' +
          '- @tensorflow/tfjs (for pure JavaScript fallback)'
        );
      }
    }
  }
}