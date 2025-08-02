/**
 * Node.js-specific TensorFlow.js loader
 * 
 * This module attempts to load the best available TensorFlow.js backend
 * for Node.js environments, with automatic fallback.
 */

export async function loadTensorFlow() {
  // Try GPU backend first
  try {
    // Use require.resolve to check if module exists before importing
    require.resolve('@tensorflow/tfjs-node-gpu');
    const tfGpu = await import('@tensorflow/tfjs-node-gpu' as string);
    console.log('Using TensorFlow.js GPU backend');
    return tfGpu as typeof import('@tensorflow/tfjs');
  } catch (gpuError) {
    // GPU backend not available, try CPU backend
    try {
      require.resolve('@tensorflow/tfjs-node');
      const tfNode = await import('@tensorflow/tfjs-node');
      console.log('Using TensorFlow.js Node.js CPU backend');
      return tfNode as typeof import('@tensorflow/tfjs');
    } catch (nodeError) {
      // Neither Node backend available, fall back to pure JS
      try {
        require.resolve('@tensorflow/tfjs');
        const tfJs = await import('@tensorflow/tfjs');
        console.log('Using TensorFlow.js pure JavaScript backend (slower performance)');
        return tfJs as typeof import('@tensorflow/tfjs');
      } catch (jsError) {
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