/**
 * Cross-platform TensorFlow.js import helper for tests.
 * 
 * This module automatically selects the appropriate TensorFlow.js backend
 * based on platform compatibility detected during Jest setup.
 */

// Re-export from @tensorflow/tfjs-core for type compatibility
export * from '@tensorflow/tfjs-core';

// Dynamically load the appropriate TensorFlow.js module
let tf: any;

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

// Also export specific properties that might be on the loaded module
export const io = tf.io;
export const version = tf.version;
export const data = tf.data;

// Default export for backward compatibility
export default tf;