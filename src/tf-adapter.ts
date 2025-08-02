/**
 * TensorFlow.js adapter module
 * 
 * This module provides a platform-agnostic interface to TensorFlow.js,
 * allowing the library to work in both Node.js and browser environments.
 * 
 * For now, it simply re-exports everything from the Node.js backend,
 * but it will be extended to support dynamic backend selection.
 */

// TODO: In phase 2, this will be replaced with dynamic imports and backend detection
import * as tf from '@tensorflow/tfjs-node';

export default tf;
export * from '@tensorflow/tfjs-node';