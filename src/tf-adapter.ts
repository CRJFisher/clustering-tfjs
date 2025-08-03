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
  tf = require('@tensorflow/tfjs');
} else {
  try {
    // Use Node.js backend for better performance
    tf = require('@tensorflow/tfjs-node');
  } catch (error) {
    // Fallback to pure JS if tfjs-node fails to load
    console.warn('tf-adapter: Failed to load @tensorflow/tfjs-node, using pure JS fallback');
    tf = require('@tensorflow/tfjs');
  }
}

export default tf;
export * from '@tensorflow/tfjs-node';