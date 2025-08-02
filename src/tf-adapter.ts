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

// For backward compatibility, continue using tfjs-node directly
// This will be replaced in Phase 3 with proper build configuration
import * as tf from '@tensorflow/tfjs-node';

export default tf;
export * from '@tensorflow/tfjs-node';