/**
 * Browser-specific TensorFlow.js loader
 * 
 * This module loads TensorFlow.js for browser environments.
 * Users must install @tensorflow/tfjs as a peer dependency.
 */

export async function loadTensorFlow() {
  // In browser environment, TensorFlow.js is expected to be loaded as a global
  if (typeof window !== 'undefined' && (window as Window & { tf?: unknown }).tf) {
    return (window as Window & { tf: typeof import('@tensorflow/tfjs') }).tf;
  }
  
  // If not available as global, try dynamic import
  try {
    const tf = await import('@tensorflow/tfjs');
    return tf as typeof import('@tensorflow/tfjs');
  } catch (error) {
    throw new Error(
      'TensorFlow.js not found. Please load it before using this library:\n' +
      '<script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs"></script>'
    );
  }
}