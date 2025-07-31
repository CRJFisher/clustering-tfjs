/**
 * Cross-platform TensorFlow.js loader for the library.
 * 
 * This module attempts to load @tensorflow/tfjs-node for better performance,
 * but falls back to @tensorflow/tfjs if native bindings are not available.
 * This ensures the library works on all platforms including Windows CI.
 */

let tf: typeof import('@tensorflow/tfjs-node');
let backendType: 'node' | 'cpu' = 'node';

try {
  // Try to load the Node.js backend with native bindings
  tf = require('@tensorflow/tfjs-node');
  backendType = 'node';
} catch (error) {
  // Fall back to CPU-only backend if native bindings fail
  try {
    tf = require('@tensorflow/tfjs');
    backendType = 'cpu';
    // Only log in development/test environments, not in production
    if (process.env.NODE_ENV !== 'production') {
      console.warn('TensorFlow.js: Using CPU backend (tfjs-node unavailable)');
    }
  } catch (fallbackError) {
    throw new Error(
      'Failed to load TensorFlow.js. Please install either @tensorflow/tfjs-node or @tensorflow/tfjs'
    );
  }
}

export default tf;
export const tfBackendType = backendType;

// Re-export all TensorFlow types and functions
export * from '@tensorflow/tfjs-node';