import * as tf from "@tensorflow/tfjs-node";

/**
 * Converts a regular (nested) JavaScript array into a TensorFlow.js tensor
 * with the provided dtype (defaults to `float32`).
 *
 * The function is wrapped in `tf.tidy` to ensure that any intermediate
 * tensors that may be created by TensorFlow.js during conversion are
 * automatically disposed of.
 */
export function arrayToTensor(
  arr: tf.TensorLike,
  dtype: tf.DataType = "float32",
): tf.Tensor {
  return tf.tidy(() => tf.tensor(arr, undefined, dtype));
}

/**
 * Converts a tensor back to a JavaScript array (synchronously).
 *
 * The returned value is a *copy* of the underlying data, so further
 * manipulations will not affect the original tensor.
 */
export function tensorToArray(tensor: tf.Tensor): any {
  // Using .arraySync() is safe here because callers explicitly request the
  // data as a JS structure. For large tensors prefer the async variant.
  return tensor.arraySync();
}

/* ------------------------------------------------------------------------- */
/*                        Distance / Similarity Metrics                      */
/* ------------------------------------------------------------------------- */

/**
 * Computes the element-wise Euclidean (ℓ2) distance between two tensors along
 * their last dimension.
 *
 * Both inputs must be broadcast-compatible. The result will have the broadcast
 * shape of `tf.sub(a, b).sum(-1)` (i.e. the shapes minus the last dimension).
 *
 * Example:
 * ```ts
 * const a = tf.tensor([[0, 0], [1, 1]]); // (2, 2)
 * const b = tf.tensor([1, 0]);           // (2)
 * euclideanDistance(a, b)  // => Tensor([1, 1])
 * ```
 */
export function euclideanDistance(a: tf.Tensor, b: tf.Tensor): tf.Tensor {
  return tf.tidy(() => tf.sub(a, b).square().sum(-1).sqrt());
}

/**
 * Computes the Manhattan (ℓ1) distance between two tensors along their last
 * dimension.
 */
export function manhattanDistance(a: tf.Tensor, b: tf.Tensor): tf.Tensor {
  return tf.tidy(() => a.sub(b).abs().sum(-1));
}

/**
 * Computes the cosine distance (1 ‑ cosine similarity) between two tensors
 * along their last dimension.
 */
export function cosineDistance(a: tf.Tensor, b: tf.Tensor): tf.Tensor {
  return tf.tidy(() => {
    const aNorm = a.norm();
    const bNorm = b.norm();
    const dot = a.mul(b).sum(-1);
    const eps = tf.scalar(1e-8);
    const denom = aNorm.mul(bNorm).add(eps);
    const similarity = dot.div(denom);
    return tf.scalar(1).sub(similarity);
  });
}

/* ------------------------------------------------------------------------- */
/*                            Broadcast Utilities                            */
/* ------------------------------------------------------------------------- */

/**
 * Efficiently computes pairwise Euclidean distance matrix for a set of points
 * represented by a 2-D tensor of shape `(n, d)`.
 *
 * The implementation uses the well-known trick
 * ‖x − y‖² = ‖x‖² + ‖y‖² − 2·xᵀy with broadcasting to avoid allocating an
 * `(n, n, d)` intermediate tensor.
 */
export function pairwiseEuclideanMatrix(points: tf.Tensor2D): tf.Tensor2D {
  return tf.tidy(() => {
    const squaredNorms = points.square().sum(1).reshape([-1, 1]); // (n,1)
    const gram = points.matMul(points.transpose()); // (n,n)
    // Using (a + bᵀ - 2G) and taking sqrt.
    const distancesSquared = squaredNorms
      .add(squaredNorms.transpose())
      .sub(gram.mul(2));

    // Numerical stability: max(dist², 0)
    const zero = tf.scalar(0, "float32");
    const distancesSquaredClamped = tf.maximum(distancesSquared, zero);
    return distancesSquaredClamped.sqrt() as tf.Tensor2D;
  });
}
