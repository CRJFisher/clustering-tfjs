/**
 * Cross-platform TensorFlow.js import helper for tests.
 * 
 * This module automatically selects the appropriate TensorFlow.js backend
 * based on platform compatibility detected during Jest setup.
 */

let tf: typeof import('@tensorflow/tfjs-node');

if (process.env.TF_FALLBACK_MODE === 'true') {
  // Use CPU-only backend when tfjs-node is not available (e.g., Windows CI)
  tf = require('@tensorflow/tfjs');
} else {
  // Use Node.js backend (preferred for performance)
  tf = require('@tensorflow/tfjs-node');
}

export default tf;
export * from '@tensorflow/tfjs-node';