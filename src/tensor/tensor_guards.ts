/**
 * Utility functions for working with tensors across different environments
 */

import * as tf from '../backend/adapter';

/**
 * Check if a value is a TensorFlow.js tensor
 * Works across different environments where tf.Tensor might not be directly available
 */
export function is_tensor(value: unknown): value is tf.Tensor {
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
export function is_tensor_2d(value: unknown): value is tf.Tensor2D {
  return is_tensor(value) && (value as tf.Tensor).rank === 2;
}

