import * as tf from '@tensorflow/tfjs-node';

/**
 * Creates the constant eigenvector for connected graphs in spectral clustering.
 *
 * For a connected graph, the smallest eigenvalue of the normalized Laplacian is 0,
 * and its corresponding eigenvector should be constant. However, numerical computation
 * can introduce small variations. sklearn replaces this with the theoretical constant
 * eigenvector to improve clustering stability.
 *
 * sklearn uses a simple constant vector 1/sqrt(n) for all entries, not the
 * degree-weighted version. This is then scaled by the spectral embedding normalization.
 *
 * @param affinity The affinity matrix
 * @returns The constant eigenvector as a column vector (n x 1)
 */
export function createConstantEigenvector(affinity: tf.Tensor2D): tf.Tensor2D {
  return tf.tidy(() => {
    const n = affinity.shape[0];

    // Create simple constant vector: all entries are 1/sqrt(n)
    // This gives a unit-norm vector with all equal entries
    const value = 1.0 / Math.sqrt(n);
    const constantVec = tf.fill([n, 1], value) as tf.Tensor2D;

    return constantVec;
  });
}
