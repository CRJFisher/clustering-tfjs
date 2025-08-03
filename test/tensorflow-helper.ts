/**
 * Cross-platform TensorFlow.js import helper for tests.
 * 
 * This module automatically selects the appropriate TensorFlow.js backend
 * based on platform compatibility detected during Jest setup.
 */

let tf: typeof import('@tensorflow/tfjs-node');

// Check if we should force CPU backend (Windows CI or tfjs-node load failure)
if (process.env.TF_FORCE_CPU_BACKEND === 'true') {
  // Use CPU-only backend when tfjs-node is not available (e.g., Windows CI)
  tf = require('@tensorflow/tfjs');
} else {
  try {
    // Try to use Node.js backend (preferred for performance)
    tf = require('@tensorflow/tfjs-node');
  } catch (error) {
    // Fallback to CPU backend if tfjs-node fails to load
    console.warn('tensorflow-helper: Falling back to CPU backend due to:', error instanceof Error ? error.message : String(error));
    tf = require('@tensorflow/tfjs');
  }
}

// Re-export all TensorFlow.js functions
export * from '@tensorflow/tfjs-core';
export default tf;