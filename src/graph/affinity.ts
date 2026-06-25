import * as tf from '../backend/adapter';

import {
  pairwise_euclidean_matrix,
  pairwise_distance_matrix,
} from '../distance/pairwise_distance';
import {
  SparseMatrix,
  sparse_matrix_from_row_maps,
  sparse_to_dense_tensor,
} from './sparse';

/**
 *  A[i, j] = exp(-gamma * ||x_i - x_j||^2)
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

    const distances = pairwise_euclidean_matrix(points);
    const sq = distances.square();

    const A = sq.mul(-gamma_val).exp() as tf.Tensor2D;

    // Average with transpose to eliminate any float32 numerical asymmetry.
    const sym = A.add(A.transpose()).div(2);
    const eye = tf.eye(sym.shape[0]);
    return sym.mul(tf.scalar(1).sub(eye)).add(eye) as tf.Tensor2D;
  });
}

/**
 * Self-loops are included by default to ensure connectivity (sklearn behaviour).
 * Symmetrised via `0.5 * (A + Aᵀ)`: mutual edges → 1.0, asymmetric → 0.5.
 *
 * Returns a dense tensor because downstream ops (eigen-decomposition) expect
 * dense input; use `compute_sparse_knn_affinity` for the CSR form.
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
 * Symmetrised as `0.5 * (A + Aᵀ)`, matching sklearn's SpectralClustering
 * connectivity path.
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
  // The squared-norm reduction runs in a tidy so its `.square()` intermediate
  // is disposed; only the final (n,) vector is kept across blocks.
  const squared_norms_kept = tf.keep(
    tf.tidy(() => points_kept.square().sum(1)),
  ) as tf.Tensor1D;

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
      const block = points_kept.slice([start, 0], [b, -1]);

      // ‖x − y‖² = ‖x‖² + ‖y‖² − 2·xᵀy  (avoids building an (n,n,d) tensor)
      const block_norms = squared_norms_kept.slice([start], [b]).reshape([b, 1]);
      const all_norms_row = squared_norms_kept.reshape([1, n_samples]);
      const cross = block.matMul(points_kept.transpose());
      const dists_squared = block_norms.add(all_norms_row).sub(cross.mul(2));

      // Skip sqrt — squared distances preserve the nearest-neighbour ordering.
      const neg_dists = dists_squared.neg(); // negate so topk picks the smallest
      // include_self=false needs k+1 candidates to drop self after sorting
      const top_k = include_self ? k : k + 1;
      const { indices } = tf.topk(neg_dists, top_k);

      // Sort ascending for deterministic tie-breaking, mirroring NumPy.
      const ind_arr = indices.arraySync() as number[][];

      for (let i = 0; i < b; i++) {
        const row_global = start + i;
        ind_arr[i].sort((a, b) => a - b);
        const neighbours = include_self
          ? ind_arr[i]
          : ind_arr[i].filter((idx) => idx !== row_global).slice(0, k);
        for (const nb of neighbours) {
          add_symmetrised_edge(row_global, nb);
        }
      }
    });
  }

  squared_norms_kept.dispose(); // caller owns points; only this helper's keeps need releasing

  return sparse_matrix_from_row_maps(rows, n_samples);
}

/**
 *   A[i, j] = 1 - cosine_distance(x_i, x_j)
 *
 * Natural similarity for direction-dominated, magnitude-noisy data (text
 * embeddings, TF-IDF vectors). Diagonal is forced to exactly 1.
 */
export function compute_cosine_affinity(points: tf.Tensor2D): tf.Tensor2D {
  return tf.tidy(() => {
    const distances = pairwise_distance_matrix(points, 'cosine');
    const ones = tf.ones_like(distances);
    const sim = ones.sub(distances);

    // Average with transpose to eliminate float32 numerical asymmetry.
    const sym = sim.add(sim.transpose()).div(2);
    const eye = tf.eye(sym.shape[0]);
    return sym.mul(tf.scalar(1).sub(eye)).add(eye) as tf.Tensor2D;
  });
}

