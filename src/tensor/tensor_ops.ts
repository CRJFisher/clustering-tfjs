import * as tf from '../backend/adapter';

/** Both inputs must be broadcast-compatible; reduction is along the last axis. */
export function euclidean_distance(a: tf.Tensor, b: tf.Tensor): tf.Tensor {
  return tf.tidy(() => tf.sub(a, b).square().sum(-1).sqrt());
}

export function manhattan_distance(a: tf.Tensor, b: tf.Tensor): tf.Tensor {
  return tf.tidy(() => a.sub(b).abs().sum(-1));
}

export function cosine_distance(a: tf.Tensor, b: tf.Tensor): tf.Tensor {
  return tf.tidy(() => {
    const a_norm = a.norm('euclidean', -1);
    const b_norm = b.norm('euclidean', -1);
    const dot = a.mul(b).sum(-1);
    const eps = tf.scalar(1e-8); // prevents divide-by-zero for the zero vector
    const denom = a_norm.mul(b_norm).add(eps);
    const similarity = dot.div(denom);
    return tf.scalar(1).sub(similarity);
  });
}
