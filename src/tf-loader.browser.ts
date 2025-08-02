/**
 * Browser-specific TensorFlow.js loader
 * 
 * This module loads TensorFlow.js for browser environments.
 * Users must install @tensorflow/tfjs as a peer dependency.
 */

export async function loadTensorFlow() {
  try {
    // Dynamic import for browser environment
    const tf = await import('@tensorflow/tfjs');
    
    // Return the default export or the entire module
    return tf.default || tf;
  } catch (error) {
    throw new Error(
      'Failed to load @tensorflow/tfjs. Please install it as a peer dependency:\n' +
      'npm install @tensorflow/tfjs'
    );
  }
}