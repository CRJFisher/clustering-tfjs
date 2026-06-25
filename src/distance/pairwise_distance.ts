import * as tf from '../backend/adapter';

import { manhattan_distance, cosine_distance } from '../tensor/tensor_ops';

/**
 * Optimised Euclidean pairwise distance using the identity
 * ‖x − y‖² = ‖x‖² + ‖y‖² − 2·xᵀy to avoid building an (n,n,d) tensor.
 */
export function pairwise_euclidean_matrix(points: tf.Tensor2D): tf.Tensor2D {
  return tf.tidy(() => {
    const squared_norms = points.square().sum(1).reshape([-1, 1]);
    const gram = points.matMul(points.transpose());

    const distances_squared = squared_norms
      .add(squared_norms.transpose())
      .sub(gram.mul(2));

    const zero = tf.scalar(0, 'float32');
    const distances_squared_clamped = tf.maximum(distances_squared, zero);

    const dist = distances_squared_clamped.sqrt();

    const dist_sym = dist.add(dist.transpose()).div(2);
    const n = dist_sym.shape[0];
    const mask = tf.ones([n, n], 'float32').sub(tf.eye(n));
    return dist_sym.mul(mask) as tf.Tensor2D;
  });
}

export function pairwise_distance_matrix(
  points: tf.Tensor2D,
  metric: 'euclidean' | 'manhattan' | 'cosine' = 'euclidean',
): tf.Tensor2D {
  switch (metric) {
    case 'euclidean':
      return pairwise_euclidean_matrix(points);

    case 'manhattan':
      return tf.tidy(() => {
        const n = points.shape[0];
        const expanded_a = points.expandDims(1);
        const expanded_b = points.expandDims(0);

        const dist = manhattan_distance(expanded_a, expanded_b) as tf.Tensor2D;

        const dist_sym = dist.add(dist.transpose()).div(2) as tf.Tensor2D;
        const mask = tf.ones([n, n], 'float32').sub(tf.eye(n));
        return dist_sym.mul(mask) as tf.Tensor2D;
      });

    case 'cosine':
      // Materialises an (n,n,d) intermediate — O(n²·d) memory, the practical
      // scalability ceiling of every cosine path (cosine k-means seeding,
      // spectral cosine affinity, HDBSCAN precomputed cosine, tracking). For
      // very large n, L2-normalize and feed a precomputed matrix instead.
      return tf.tidy(() => {
        const n = points.shape[0];
        const expanded_a = points.expandDims(1);
        const expanded_b = points.expandDims(0);

        const dist = cosine_distance(expanded_a, expanded_b) as tf.Tensor2D;

        const dist_sym = dist.add(dist.transpose()).div(2) as tf.Tensor2D;
        const mask = tf.ones([n, n], 'float32').sub(tf.eye(n));
        return dist_sym.mul(mask) as tf.Tensor2D;
      });

    default:
      throw new Error(`Unsupported metric '${metric}'.`);
  }
}
