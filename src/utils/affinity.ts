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

    // Default gamma mirrors scikit-learn behaviour for its RBF kernel used
    // inside SpectralClustering: gamma = 1.0 / n_features when the user does
    // not specify a value.  We align with that default to ensure parity with
    // reference fixtures.

    const gammaVal = gamma ?? 1.0 / nFeatures;

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

  const nSamples = points.shape[0];

  if (nSamples === 0) {
    throw new Error("Input points tensor must contain at least one sample.");
  }

  if (k >= nSamples) {
    throw new Error("k (nNeighbors) must be smaller than the number of samples.");
  }

  /* --------------------------------------------------------------------- */
  /*  Implementation note – memory-efficient block-wise distance scanning    */
  /* --------------------------------------------------------------------- */
  // A naive implementation constructs the full pair-wise distance matrix
  // (n×n) and then selects the k closest entries per row.  This requires
  // O(n²) memory which becomes prohibitive for large datasets.
  //
  // Instead we process the data in reasonable row-blocks: for each block of
  // b rows we compute the distances to *all* samples (b×n) which has a peak
  // memory footprint of O(b·n).  With a modest block size (e.g. 1024) this
  // scales to tens of thousands of samples while maintaining GPU/CPU
  // efficiency thanks to matrix operations.

  // Keep tensors that are required across blocks to avoid accidental disposal.
  const pointsKept = tf.keep(points) as tf.Tensor2D;
  const squaredNormsKept = tf.keep(pointsKept.square().sum(1)) as tf.Tensor1D; // (n)

  const coords: number[][] = [];

  // Empirically chosen – small enough to fit typical accelerator memory while
  // large enough to utilise BLAS throughput.
  const BLOCK_SIZE = 1024;

  for (let start = 0; start < nSamples; start += BLOCK_SIZE) {
    const b = Math.min(BLOCK_SIZE, nSamples - start);

    tf.tidy(() => {
      // Slice current block (b,d)
      const block = pointsKept.slice([start, 0], [b, -1]);

      // Efficient squared Euclidean distances using the identity
      // ‖x − y‖² = ‖x‖² + ‖y‖² − 2·xᵀy
      const blockNorms = squaredNormsKept.slice([start], [b]).reshape([b, 1]); // (b,1)
      const allNormsRow = squaredNormsKept.reshape([1, nSamples]); // (1,n)

      const cross = block.matMul(pointsKept.transpose()); // (b,n)
      const distsSquared = blockNorms.add(allNormsRow).sub(cross.mul(2)); // (b,n)

      // We can avoid the costly sqrt, distances squared preserve ordering.
      const negDists = distsSquared.neg(); // Want k smallest ⇒ largest of negative values.

      // topk on each row (b,k+1) – self-distance will always be the largest.
      const { indices } = tf.topk(negDists, k + 1);

      const indArr = indices.arraySync() as number[][];
      for (let i = 0; i < b; i++) {
        const rowGlobal = start + i;
        for (let j = 1; j < indArr[i].length; j++) {
          coords.push([rowGlobal, indArr[i][j]]);
        }
      }
    }); // tidy – dispose temporaries for this block
  }

  // Release kept tensors
  pointsKept.dispose();
  squaredNormsKept.dispose();

  if (coords.length === 0) {
    return tf.zeros([nSamples, nSamples]);
  }

  // Scatter ones into a dense zero matrix – TensorFlow.js scatterND expects
  // typed arrays / tensors for the indices as well as values.  Passing plain
  // JS arrays is fine, the backend converts them on the fly.
  return tf.tidy(() => {
    const values = tf.ones([coords.length]);
    const dense = tf.scatterND(coords, values, [nSamples, nSamples]) as tf.Tensor2D;
    // Symmetrise: A = max(A, Aᵀ)
    return tf.maximum(dense, dense.transpose()) as tf.Tensor2D;
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
