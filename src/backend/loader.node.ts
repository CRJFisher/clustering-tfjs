export async function load_tensor_flow() {
  try {
    require.resolve('@tensorflow/tfjs-node-gpu');
    // `as string` prevents TS from resolving types for this optional peer dep at build time.
    const tf_gpu = await import('@tensorflow/tfjs-node-gpu' as string);
    console.log('Using TensorFlow.js GPU backend');
    return tf_gpu as typeof import('@tensorflow/tfjs');
  } catch {
    try {
      require.resolve('@tensorflow/tfjs-node');
      const tf_node = await import('@tensorflow/tfjs-node');
      console.log('Using TensorFlow.js Node.js CPU backend');
      return tf_node as typeof import('@tensorflow/tfjs');
    } catch {
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
