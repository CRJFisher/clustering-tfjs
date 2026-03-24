/**
 * Shared Gram-Schmidt reorthogonalization utilities.
 *
 * Used by both the Lanczos solver (Float64Array basis vectors) and
 * the QR eigensolver (number[][] column-major eigenvector matrix).
 */

/**
 * Projects out components of `w` along all `basis` vectors.
 * Modifies `w` in place. Works with any numeric array type.
 */
export function reorthogonalizeVector<T extends { [i: number]: number; length: number }>(
  w: T,
  basis: T[],
  n: number,
): void {
  for (let b = 0; b < basis.length; b++) {
    const q = basis[b];
    let dot = 0;
    for (let i = 0; i < n; i++) dot += q[i] * w[i];
    for (let i = 0; i < n; i++) w[i] -= dot * q[i];
  }
}

/**
 * Modified Gram-Schmidt on columns of a row-major matrix.
 * Normalizes each column and projects it out of all subsequent columns.
 * Modifies the matrix in place.
 *
 * @param rows - Row-major matrix (rows[i][j] = element at row i, col j)
 * @param n - Number of rows (and columns)
 */
export function gramSchmidtColumns(rows: number[][], n: number): void {
  for (let col = 0; col < n; col++) {
    // Normalize column `col`
    let norm = 0;
    for (let row = 0; row < n; row++) norm += rows[row][col] * rows[row][col];
    norm = Math.sqrt(norm);
    if (norm < 1e-15) continue;
    for (let row = 0; row < n; row++) rows[row][col] /= norm;

    // Project column `col` out of all subsequent columns
    for (let other = col + 1; other < n; other++) {
      let dot = 0;
      for (let row = 0; row < n; row++) dot += rows[row][col] * rows[row][other];
      for (let row = 0; row < n; row++) rows[row][other] -= dot * rows[row][col];
    }
  }
}
