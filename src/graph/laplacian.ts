import * as tf from '../backend/adapter';
import { smallest_eigenvectors_with_values } from '../eigen/smallest_eigenvectors_with_values';
import { SparseMatrix, sparse_row_sums } from './sparse';

/* -------------------------------------------------------------------------- */
/*                        Graph Laplacian – core utilities                    */
/* -------------------------------------------------------------------------- */

/**
 * Computes the (row) degree vector for the provided *affinity / similarity*
 * matrix.
 *
 * The input must be a **square** `tf.Tensor2D` whose entries represent edge
 * weights `A[i,j]` of an undirected graph.  For *k*-NN graphs the matrix is
 * expected to be symmetrised (`A = 0.5 * (A + Aᵀ)`).
 *
 * The returned tensor is a 1-D vector where `deg[i] = Σ_j A[i,j]`.
 */
export function degree_vector(A: tf.Tensor2D): tf.Tensor1D {
  if (A.shape.length !== 2 || A.shape[0] !== A.shape[1]) {
    throw new Error('Affinity matrix must be square (n × n).');
  }

  // Sum along axis 1 (rows) – (n)
  return tf.tidy(() => A.sum(1) as tf.Tensor1D);
}

/**
 * Computes the *symmetric normalised* graph Laplacian
 *
 *     L = I  −  D^{-1/2} · A · D^{-1/2}
 *
 * where
 *   • `A` is the affinity / adjacency matrix
 *   • `D` is the diagonal *degree* matrix with `D[i,i] = Σ_j A[i,j]`
 *
 * Isolated vertices (zero degree) are handled gracefully by treating
 * `D^{-1/2}` as **0** for the corresponding diagonal entry which leaves the
 * row & column of `L` equal to the identity matrix (no connections).
 */
export function normalised_laplacian(A: tf.Tensor2D): tf.Tensor2D;
export function normalised_laplacian(
  A: tf.Tensor2D,
  return_diag: true,
): { laplacian: tf.Tensor2D; sqrt_degrees: tf.Tensor1D };
export function normalised_laplacian(
  A: tf.Tensor2D,
  return_diag = false,
): tf.Tensor2D | { laplacian: tf.Tensor2D; sqrt_degrees: tf.Tensor1D } {
  return tf.tidy(() => {
    const n = A.shape[0];

    // First, zero out the diagonal of A to match scipy behavior
    // "Diagonal entries of the input adjacency matrix are ignored and
    // replaced with zeros for the purpose of normalization"
    const diag_mask = tf.sub(1, tf.eye(n));
    const A_no_diag = A.mul(diag_mask) as tf.Tensor2D;

    const deg = degree_vector(A_no_diag); // (n)

    // d^{-1/2} – set entries with deg == 0 to 1 (for isolated nodes)
    const inv_sqrt = tf.where(
      deg.equal(0),
      tf.ones_like(deg),
      deg.pow(-0.5),
    ) as tf.Tensor1D; // (n)

    // Build outer product inv_sqrt[:,None] * inv_sqrt[None,:]  (n,n)
    const diag_col = inv_sqrt.reshape([n, 1]);
    const diag_row = inv_sqrt.reshape([1, n]);
    const scaling = diag_col.matMul(diag_row) as tf.Tensor2D; // (n,n)

    // Scale the affinity matrix (with diagonal already zeroed)
    const scaled_affinity = A_no_diag.mul(scaling) as tf.Tensor2D;

    // L = I - scaled_affinity
    // This ensures diagonal entries are exactly 1 for non-isolated nodes
    const I = tf.eye(n);
    const laplacian = I.sub(scaled_affinity) as tf.Tensor2D;

    if (return_diag) {
      // Return both Laplacian and sqrt(degrees) for eigenvector recovery
      // Note: we return inv_sqrt which is D^(-1/2), so for recovery we need to divide by it
      return {
        laplacian: laplacian,
        sqrt_degrees: inv_sqrt,
      };
    }

    return laplacian;
  });
}

export interface MatrixFreeOperator {
  n: number;
  matvec: (vector: Float64Array) => Float64Array;
}

export interface SparseNormalisedLaplacian {
  operator: MatrixFreeOperator;
  sqrt_degrees: Float64Array;
  degrees: Float64Array;
}

export function sparse_normalised_laplacian_operator(
  affinity: SparseMatrix,
): SparseNormalisedLaplacian {
  if (affinity.rows !== affinity.cols) {
    throw new Error('Affinity matrix must be square (n × n).');
  }

  const n = affinity.rows;
  const degrees = sparse_row_sums(affinity, true);
  const inv_sqrt_degrees = new Float64Array(n);

  for (let i = 0; i < n; i++) {
    inv_sqrt_degrees[i] = degrees[i] === 0 ? 1 : Math.pow(degrees[i], -0.5);
  }

  const operator: MatrixFreeOperator = {
    n,
    matvec(vector: Float64Array): Float64Array {
      if (vector.length !== n) {
        throw new Error(
          `Vector length ${vector.length} does not match Laplacian size ${n}.`,
        );
      }

      const result = new Float64Array(n);
      for (let row = 0; row < n; row++) {
        let scaled_affinity_sum = 0;
        const row_scale = inv_sqrt_degrees[row];

        for (
          let ptr = affinity.indptr[row];
          ptr < affinity.indptr[row + 1];
          ptr++
        ) {
          const col = affinity.indices[ptr];
          if (col === row) continue;
          scaled_affinity_sum +=
            affinity.data[ptr] * row_scale * inv_sqrt_degrees[col] * vector[col];
        }

        result[row] = vector[row] - scaled_affinity_sum;
      }

      return result;
    },
  };

  return {
    operator,
    sqrt_degrees: inv_sqrt_degrees,
    degrees,
  };
}

/**
 * Convenience helper that returns the `k` smallest eigenvectors of the
 * provided symmetric matrix *as a TensorFlow.js tensor* (n × k).
 *
 * Delegates to `smallest_eigenvectors_with_values` for consistent
 * solver selection (Lanczos for large matrices, Jacobi for small).
 */
export function smallest_eigenvectors(
  matrix: tf.Tensor2D,
  k: number,
): tf.Tensor2D {
  if (!Number.isInteger(k) || k < 1) {
    throw new Error('k must be a positive integer.');
  }

  const { eigenvectors, eigenvalues } = smallest_eigenvectors_with_values(matrix, k);
  eigenvalues.dispose();
  return eigenvectors;
}
