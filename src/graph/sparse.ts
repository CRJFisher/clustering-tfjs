import * as tf from '../backend/adapter';

export interface SparseMatrix {
  rows: number;
  cols: number;
  indptr: Int32Array;
  indices: Int32Array;
  data: Float64Array;
}

interface SparseMatrixStats {
  shape: [number, number];
  nnz: number;
  min: number;
  max: number;
  mean: number;
}

export function sparse_matrix_from_row_maps(
  rows: Array<Map<number, number>>,
  cols: number = rows.length,
): SparseMatrix {
  const n_rows = rows.length;
  const indptr = new Int32Array(n_rows + 1);
  let nnz = 0;

  for (let row = 0; row < n_rows; row++) {
    nnz += rows[row].size;
    indptr[row + 1] = nnz;
  }

  const indices = new Int32Array(nnz);
  const data = new Float64Array(nnz);
  let offset = 0;

  for (let row = 0; row < n_rows; row++) {
    // CSR requires column indices to be in ascending order within each row.
    const entries = Array.from(rows[row].entries()).sort((a, b) => a[0] - b[0]);
    for (const [col, value] of entries) {
      if (col < 0 || col >= cols) {
        throw new Error(`Sparse column index ${col} is outside matrix width ${cols}.`);
      }
      indices[offset] = col;
      data[offset] = value;
      offset += 1;
    }
  }

  return {
    rows: n_rows,
    cols,
    indptr,
    indices,
    data,
  };
}

export function sparse_to_dense_array(matrix: SparseMatrix): number[][] {
  const dense = Array.from({ length: matrix.rows }, () =>
    new Array(matrix.cols).fill(0),
  );

  for (let row = 0; row < matrix.rows; row++) {
    for (let ptr = matrix.indptr[row]; ptr < matrix.indptr[row + 1]; ptr++) {
      dense[row][matrix.indices[ptr]] = matrix.data[ptr];
    }
  }

  return dense;
}

export function sparse_to_dense_tensor(matrix: SparseMatrix): tf.Tensor2D {
  return tf.tensor2d(
    sparse_to_dense_array(matrix),
    [matrix.rows, matrix.cols],
    'float32',
  );
}

export function sparse_row_sums(
  matrix: SparseMatrix,
  ignore_diagonal: boolean = false,
): Float64Array {
  const sums = new Float64Array(matrix.rows);

  for (let row = 0; row < matrix.rows; row++) {
    let sum = 0;
    for (let ptr = matrix.indptr[row]; ptr < matrix.indptr[row + 1]; ptr++) {
      const col = matrix.indices[ptr];
      if (ignore_diagonal && col === row) continue;
      sum += matrix.data[ptr];
    }
    sums[row] = sum;
  }

  return sums;
}

export function sparse_matvec(
  matrix: SparseMatrix,
  vector: Float64Array,
): Float64Array {
  if (vector.length !== matrix.cols) {
    throw new Error(
      `Vector length ${vector.length} does not match sparse matrix width ${matrix.cols}.`,
    );
  }

  const result = new Float64Array(matrix.rows);
  for (let row = 0; row < matrix.rows; row++) {
    let sum = 0;
    for (let ptr = matrix.indptr[row]; ptr < matrix.indptr[row + 1]; ptr++) {
      sum += matrix.data[ptr] * vector[matrix.indices[ptr]];
    }
    result[row] = sum;
  }

  return result;
}

export function sparse_stats(matrix: SparseMatrix): SparseMatrixStats {
  const total_entries = matrix.rows * matrix.cols;
  let nonzero_sum = 0;
  // When fully dense every entry is stored so Infinity lets the loop find the real min.
  // When sparse, implicit zeros exist and 0 is already the floor.
  let min = matrix.data.length === total_entries ? Infinity : 0;
  // When no entries are stored, the matrix is all implicit zeros so max is 0.
  let max = matrix.data.length === 0 ? 0 : -Infinity;

  for (const value of matrix.data) {
    min = Math.min(min, value);
    max = Math.max(max, value);
    nonzero_sum += value;
  }

  if (matrix.data.length === 0) {
    min = 0;
  }

  return {
    shape: [matrix.rows, matrix.cols],
    nnz: matrix.data.length,
    min,
    max,
    mean: total_entries === 0 ? 0 : nonzero_sum / total_entries,
  };
}
