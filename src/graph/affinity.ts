import * as tf from '../backend/adapter';

import { pairwise_euclidean_matrix } from '../distance/pairwise_distance';
import {
  SparseMatrix,
  sparse_matrix_from_row_maps,
  sparse_to_dense_tensor,
} from './sparse';

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
    const n_features = points.shape[1];

    // Default gamma mirrors scikit-learn behaviour for its RBF kernel used
    // inside SpectralClustering: gamma = 1.0 / n_features when the user does
    // not specify a value.  We align with that default to ensure parity with
    // reference fixtures.

    const gamma_val = gamma ?? 1.0 / n_features;

    const distances = pairwise_euclidean_matrix(points); // (n, n)

    // squared distances
    const sq = distances.square();

    const A = sq.mul(-gamma_val).exp() as tf.Tensor2D;

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
 * For each sample the `k` closest neighbours are connected with affinity
 * value **1**. Self-loops are included to ensure connectivity, matching
 * sklearn's behavior. The final matrix is **symmetrised** via `0.5 * (A + Aᵀ)`.
 * Mutual edges get weight 1.0 while asymmetric edges get weight 0.5.
 *
 * The result is returned as a dense `tf.Tensor2D` containing zeros for
 * non-connected pairs.  While a sparse representation would be more memory
 * efficient, downstream TensorFlow.js ops (e.g. eigen-decomposition) currently
 * expect dense tensors.
 */
export function compute_knn_affinity(
  points: tf.Tensor2D,
  k: number,
  include_self: boolean = true,
): tf.Tensor2D {
  return sparse_to_dense_tensor(
    compute_sparse_knn_affinity(points, k, include_self),
  );
}

/**
 * Builds a sparse k-nearest-neighbour connectivity graph in CSR form.
 *
 * The directed kNN graph is symmetrised as `0.5 * (A + Aᵀ)`, matching the
 * existing dense helper and sklearn's SpectralClustering connectivity path.
 */
export function compute_sparse_knn_affinity(
  points: tf.Tensor2D,
  k: number,
  include_self: boolean = true,
): SparseMatrix {
  if (!Number.isInteger(k) || k < 1) {
    throw new Error('k (n_neighbors) must be a positive integer.');
  }

  const n_samples = points.shape[0];

  if (n_samples === 0) {
    throw new Error('Input points tensor must contain at least one sample.');
  }

  if (k >= n_samples) {
    throw new Error(
      'k (n_neighbors) must be smaller than the number of samples.',
    );
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
  const points_kept = tf.keep(points) as tf.Tensor2D;
  const squared_norms_kept = tf.keep(points_kept.square().sum(1)) as tf.Tensor1D; // (n)

  const rows: Array<Map<number, number>> = Array.from(
    { length: n_samples },
    () => new Map<number, number>(),
  );

  const add_symmetrised_edge = (row: number, col: number): void => {
    if (row === col) {
      rows[row].set(col, 1);
      return;
    }

    rows[row].set(col, (rows[row].get(col) ?? 0) + 0.5);
    rows[col].set(row, (rows[col].get(row) ?? 0) + 0.5);
  };

  // Empirically chosen – small enough to fit typical accelerator memory while
  // large enough to utilise BLAS throughput.
  const BLOCK_SIZE = 1024;

  for (let start = 0; start < n_samples; start += BLOCK_SIZE) {
    const b = Math.min(BLOCK_SIZE, n_samples - start);

    tf.tidy(() => {
      // Slice current block (b,d)
      const block = points_kept.slice([start, 0], [b, -1]);

      // Efficient squared Euclidean distances using the identity
      // ‖x − y‖² = ‖x‖² + ‖y‖² − 2·xᵀy
      const block_norms = squared_norms_kept.slice([start], [b]).reshape([b, 1]); // (b,1)
      const all_norms_row = squared_norms_kept.reshape([1, n_samples]); // (1,n)

      const cross = block.matMul(points_kept.transpose()); // (b,n)
      const dists_squared = block_norms.add(all_norms_row).sub(cross.mul(2)); // (b,n)

      // We can avoid the costly sqrt, distances squared preserve ordering.
      const neg_dists = dists_squared.neg(); // Want k smallest ⇒ largest of negative values.

      // topk on each row
      // When include_self=true, k neighbors include self
      // When include_self=false, we need k+1 to later filter out self
      const top_k = include_self ? k : k + 1;
      const { indices } = tf.topk(neg_dists, top_k);

      // Collect indices and apply deterministic tie-breaking: sort ascending
      // so that ties are resolved towards the lower index mirroring NumPy.
      const ind_arr = indices.arraySync() as number[][];

      for (let i = 0; i < b; i++) {
        const row_global = start + i;

        // Sort to achieve deterministic order of equal-distance neighbours.
        ind_arr[i].sort((a, b) => a - b);

        let neighbours: number[];
        if (include_self) {
          // When include_self=true, the k neighbors already include self
          neighbours = ind_arr[i];
        } else {
          // Remove self-index to get exactly k neighbors (excluding self)
          neighbours = ind_arr[i].filter((idx) => idx !== row_global).slice(0, k);
        }

        for (const nb of neighbours) {
          add_symmetrised_edge(row_global, nb);
        }
      }
    }); // tidy – dispose temporaries for this block
  }

  // Release tensors created by this helper. The input tensor is caller-owned.
  squared_norms_kept.dispose();

  return sparse_matrix_from_row_maps(rows, n_samples);
}

/**
 * Convenience wrapper that dispatches to the appropriate affinity builder
 * based on the provided `affinity` option.
 */
export function compute_affinity_matrix(
  points: tf.Tensor2D,
  options:
    | { affinity: 'rbf'; gamma?: number }
    | { affinity: 'nearest_neighbors'; n_neighbors: number },
): tf.Tensor2D {
  if (options.affinity === 'rbf') {
    return compute_rbf_affinity(points, options.gamma);
  }

  // nearest neighbours - include self-loops for connectivity
  return compute_knn_affinity(points, options.n_neighbors, true);
}
