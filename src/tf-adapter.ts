/**
 * TensorFlow.js adapter module
 * 
 * This module provides a platform-agnostic interface to TensorFlow.js,
 * allowing the library to work in both Node.js and browser environments.
 * 
 * Phase 2 implementation: For now, we maintain backward compatibility
 * by still importing tfjs-node directly, but the infrastructure for
 * dynamic loading is ready in tf-backend.ts and loaders.
 */

// Handle Windows CI environment where native modules fail
let tf: typeof import('@tensorflow/tfjs-node');

if (process.platform === 'win32' && process.env.CI) {
  // Use pure JS implementation on Windows CI
  try {
    tf = require('@tensorflow/tfjs');
  } catch (error) {
    console.error('tf-adapter: Failed to load @tensorflow/tfjs on Windows CI');
    throw new Error(`Failed to load TensorFlow.js: ${error instanceof Error ? error.message : String(error)}`);
  }
} else {
  try {
    // Use Node.js backend for better performance
    tf = require('@tensorflow/tfjs-node');
  } catch (error) {
    // Fallback to pure JS if tfjs-node fails to load
    console.warn('tf-adapter: Failed to load @tensorflow/tfjs-node, using pure JS fallback');
    try {
      tf = require('@tensorflow/tfjs');
    } catch (fallbackError) {
      console.error('tf-adapter: Failed to load @tensorflow/tfjs fallback');
      throw new Error(`Failed to load TensorFlow.js: ${fallbackError instanceof Error ? fallbackError.message : String(fallbackError)}`);
    }
  }
}

export default tf;
// Re-export everything from the loaded module
export * from '@tensorflow/tfjs-core';