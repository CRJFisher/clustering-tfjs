import * as tf from "@tensorflow/tfjs-node";

import { manhattanDistance, cosineDistance } from "./tensor";

/**
 * Optimised Euclidean pairwise distance using the identity
 * ‖x − y‖² = ‖x‖² + ‖y‖² − 2·xᵀy to avoid building an (n,n,d) tensor.
 */
export function pairwiseEuclideanMatrix(points: tf.Tensor2D): tf.Tensor2D {
  return tf.tidy(() => {
    const squaredNorms = points.square().sum(1).reshape([-1, 1]); // (n,1)
    const gram = points.matMul(points.transpose()); // (n,n)

    const distancesSquared = squaredNorms
      .add(squaredNorms.transpose())
      .sub(gram.mul(2));

    const zero = tf.scalar(0, "float32");
    const distancesSquaredClamped = tf.maximum(distancesSquared, zero);

    const dist = distancesSquaredClamped.sqrt();

    const distSym = dist.add(dist.transpose()).div(2);
    const n = distSym.shape[0];
    const mask = tf.ones([n, n], "float32").sub(tf.eye(n));
    return distSym.mul(mask) as tf.Tensor2D;
  });
}

/**
 * Computes the pairwise distance matrix for the given points according to the
 * requested metric.
 *
 * The result is an `(n, n)` tensor `D` where `D[i, j]` contains the distance
 * between row `i` and row `j` of the input `points`.
 *
 * Supported metrics:
 *   • "euclidean"  – ℓ2 distance (uses an optimised implementation)
 *   • "manhattan"  – ℓ1 distance
 *   • "cosine"     – 1 − cosine-similarity
 *
 * For performance and numerical stability the computation is wrapped in
 * `tf.tidy` so that all intermediate tensors are eagerly disposed.
 */
export function pairwiseDistanceMatrix(
  points: tf.Tensor2D,
  metric: "euclidean" | "manhattan" | "cosine" = "euclidean",
): tf.Tensor2D {
  switch (metric) {
    case "euclidean":
      return pairwiseEuclideanMatrix(points);

    case "manhattan":
      return tf.tidy(() => {
        const n = points.shape[0];
        const expandedA = points.expandDims(1); // (n,1,d)
        const expandedB = points.expandDims(0); // (1,n,d)

        const dist = manhattanDistance(expandedA, expandedB) as tf.Tensor2D;

        const distSym = dist.add(dist.transpose()).div(2) as tf.Tensor2D;
        const mask = tf.ones([n, n], "float32").sub(tf.eye(n));
        return distSym.mul(mask) as tf.Tensor2D;
      });

    case "cosine":
      return tf.tidy(() => {
        const n = points.shape[0];
        const expandedA = points.expandDims(1); // (n,1,d)
        const expandedB = points.expandDims(0); // (1,n,d)

        const dist = cosineDistance(expandedA, expandedB) as tf.Tensor2D;

        const distSym = dist.add(dist.transpose()).div(2) as tf.Tensor2D;
        const mask = tf.ones([n, n], "float32").sub(tf.eye(n));
        return distSym.mul(mask) as tf.Tensor2D;
      });

    default:
      // eslint-disable-next-line @typescript-eslint/restrict-template-expressions
      throw new Error(`Unsupported metric '${metric}'.`);
  }
}
