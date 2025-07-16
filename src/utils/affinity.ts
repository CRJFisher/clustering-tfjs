import * as tf from "@tensorflow/tfjs-node";

import { pairwiseEuclideanMatrix } from "./pairwise_distance";

/**
 * Computes the RBF (Gaussian) kernel affinity matrix for the given points.
 *
 *  A[i, j] = exp(-gamma * ||x_i - x_j||^2)
 *
 *  • The diagonal is guaranteed to be exactly 1 (because the distance is 0).
 *  • The result is symmetric by construction.
 *
 * The function is wrapped in `tf.tidy` so that all intermediate tensors are
 * automatically disposed of once the result tensor has been returned.
 */
export function compute_rbf_affinity(
  points: tf.Tensor2D,
  gamma?: number,
): tf.Tensor2D {
  return tf.tidy(() => {
    const nFeatures = points.shape[1];

    // Default gamma mirrors scikit-learn: 1 / n_features
    const gammaVal = gamma ?? 1 / nFeatures;

    const distances = pairwiseEuclideanMatrix(points); // (n, n)

    // squared distances
    const sq = distances.square();

    const A = sq.mul(-gammaVal).exp() as tf.Tensor2D;

    // Ensure exact symmetry by averaging with its transpose (to mitigate any
    // potential numerical asymmetry) and set the diagonal to 1.
    const sym = A.add(A.transpose()).div(2);
    const eye = tf.eye(sym.shape[0]);
    return sym.mul(tf.scalar(1).sub(eye)).add(eye) as tf.Tensor2D;
  });
}

/**
 * Builds a (k-)nearest-neighbour adjacency / affinity matrix.
 *
 * For each sample the `k` closest neighbours (excluding itself) are connected
 * with affinity value **1**. The final matrix is **symmetrised** via
 * `max(A, Aᵀ)` so that an edge is present when either sample appears in the
 * other's neighbourhood.
 *
 * The result is returned as a dense `tf.Tensor2D` containing zeros for
 * non-connected pairs.  While a sparse representation would be more memory
 * efficient, downstream TensorFlow.js ops (e.g. eigen-decomposition) currently
 * expect dense tensors.
 */
export function compute_knn_affinity(
  points: tf.Tensor2D,
  k: number,
): tf.Tensor2D {
  if (!Number.isInteger(k) || k < 1) {
    throw new Error("k (nNeighbors) must be a positive integer.");
  }

  return tf.tidy(() => {
    const nSamples = points.shape[0];

    // Compute full distance matrix (n, n)
    const dist = pairwiseEuclideanMatrix(points);

    // We want the k *smallest* distances, so we invert the sign and use tf.topk
    // which selects the k *largest* elements.
    const { indices } = tf.topk(dist.neg(), k + 1); // +1 to account for the self-distance 0

    // Build coordinates for edges excluding the first index in each row which
    // corresponds to the sample itself (distance 0).
    const coords: number[][] = [];
    const indArr = indices.arraySync() as number[][];
    for (let i = 0; i < nSamples; i++) {
      for (let j = 1; j < indArr[i].length; j++) {
        const nb = indArr[i][j];
        coords.push([i, nb]);
      }
    }

    if (coords.length === 0) {
      // Degenerate case when nSamples == 1
      return tf.zeros([nSamples, nSamples]);
    }

    // Values for connected edges (all ones)
    const values = tf.ones([coords.length]);

    // Create sparse representation then densify.  TensorFlow.js currently has
    // experimental support for sparse tensors. We fallback to scatterND into a
    // dense zero matrix which is supported in the core API.
    const dense = tf.scatterND(coords, values, [nSamples, nSamples]) as tf.Tensor2D;

    // Symmetrise: A = max(A, Aᵀ)
    const sym = tf.maximum(dense, dense.transpose()) as tf.Tensor2D;

    return sym;
  });
}

/**
 * Convenience wrapper that dispatches to the appropriate affinity builder
 * based on the provided `affinity` option.
 */
export function compute_affinity_matrix(
  points: tf.Tensor2D,
  options: { affinity: "rbf"; gamma?: number } | { affinity: "nearest_neighbors"; nNeighbors: number },
): tf.Tensor2D {
  if (options.affinity === "rbf") {
    return compute_rbf_affinity(points, options.gamma);
  }

  // nearest neighbours
  return compute_knn_affinity(points, options.nNeighbors);
}

