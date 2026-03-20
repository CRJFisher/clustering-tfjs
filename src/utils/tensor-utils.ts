/**
 * Utility functions for working with tensors across different environments
 */

import * as tf from '../tf-adapter';

/**
 * Check if a value is a TensorFlow.js tensor
 * Works across different environments where tf.Tensor might not be directly available
 */
export function isTensor(value: unknown): value is tf.Tensor {
  if (!value || typeof value !== 'object') {
    return false;
  }
  
  // Check for tensor-like properties
  const obj = value as Record<string, unknown>;
  return (
    typeof obj.dtype === 'string' &&
    typeof obj.shape === 'object' &&
    Array.isArray(obj.shape) &&
    typeof obj.rank === 'number' &&
    typeof obj.dataSync === 'function' &&
    typeof obj.dispose === 'function'
  );
}

/**
 * Check if a value is a 2D tensor
 */
export function isTensor2D(value: unknown): value is tf.Tensor2D {
  return isTensor(value) && (value as tf.Tensor).rank === 2;
}

/**
 * Converts a DataMatrix (Tensor2D or number[][]) to a tf.Tensor2D.
 * If the input is already a Tensor2D, returns it directly (no copy).
 * If the input is a number[][], creates a new tensor.
 *
 * @param X - Input data matrix
 * @returns The data as a tf.Tensor2D
 */
export function toTensor2D(X: tf.Tensor2D | number[][]): tf.Tensor2D {
  return isTensor(X) ? (X as tf.Tensor2D) : tf.tensor2d(X as number[][]);
}