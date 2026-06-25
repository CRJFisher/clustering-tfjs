import * as tf from '../backend/adapter';
import { SparseMatrix, sparse_row_sums } from './sparse';

/**
 * For *k*-NN graphs the affinity matrix must be symmetrised
 * (`A = 0.5 * (A + Aᵀ)`) before calling this function.
 */
export function degree_vector(A: tf.Tensor2D): tf.Tensor1D {
  if (A.shape.length !== 2 || A.shape[0] !== A.shape[1]) {
    throw new Error('Affinity matrix must be square (n × n).');
  }

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

    // "Diagonal entries of the input adjacency matrix are ignored and
    // replaced with zeros for the purpose of normalization" (scipy parity)
    const diag_mask = tf.sub(1, tf.eye(n));
    const A_no_diag = A.mul(diag_mask) as tf.Tensor2D;

    const deg = degree_vector(A_no_diag); // (n)

    const inv_sqrt = tf.where(
      deg.equal(0),
      tf.ones_like(deg),
      deg.pow(-0.5),
    ) as tf.Tensor1D; // (n)

    const diag_col = inv_sqrt.reshape([n, 1]);
    const diag_row = inv_sqrt.reshape([1, n]);
    const scaling = diag_col.matMul(diag_row) as tf.Tensor2D; // (n,n)

    const scaled_affinity = A_no_diag.mul(scaling) as tf.Tensor2D;

    const I = tf.eye(n);
    const laplacian = I.sub(scaled_affinity) as tf.Tensor2D;

    if (return_diag) {
      // sqrt_degrees holds D^{-1/2}, not D^{1/2} — callers must divide by it, not multiply
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

