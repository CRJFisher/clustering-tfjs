import * as tf from '../backend/adapter';

/**
 * Structural duck-type check so tensors from a foreign tf.js build
 * (where `instanceof tf.Tensor` would fail) are still recognised.
 */
export function is_tensor(value: unknown): value is tf.Tensor {
  if (!value || typeof value !== 'object') {
    return false;
  }
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

