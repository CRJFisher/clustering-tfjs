/**
 * Many numerical eigensolvers return eigenvectors in arbitrary order and with
 * an arbitrary global ± sign per vector.  For downstream algorithms (e.g.
 * Spectral Clustering) we require a stable ordering and sign convention so
 * that identical input always produces identical embeddings.
 *
 * The convention implemented here matches scikit-learn:
 *   1. Eigen-pairs are sorted by ascending eigen-value.
 *   2. For every eigen-vector the component with the largest absolute value
 *      is made positive by optionally multiplying the vector by −1.
 *
 * The function operates purely on native JavaScript arrays to avoid pulling
 * TensorFlow.js into low-level utilities and reduce garbage generation.
 */

export interface EigenPairInput {
  eigenvalues: number[];
  eigenvectors: number[][]; // shape (n, m) – column j belongs to eigenvalue j (m may equal n or k)
}

export interface EigenPairOutput {
  eigenvalues: number[];
  eigenvectors: number[][];
}

export function deterministic_eigenpair_processing(
  input: EigenPairInput,
): EigenPairOutput {
  const { eigenvalues, eigenvectors } = input;

  if (eigenvectors.length === 0) {
    return {
      eigenvalues: [],
      eigenvectors: [],
    };
  }

  const n = eigenvectors.length;
  const m = eigenvalues.length;

  if (eigenvectors.some((row) => row.length !== m)) {
    throw new Error(
      `eigenvectors column count must match eigenvalues length (${m}).`,
    );
  }

  const indexed = eigenvalues.map((val, idx) => ({ val, idx }));
  indexed.sort((a, b) => a.val - b.val);

  const eigenvalues_sorted: number[] = indexed.map((p) => p.val);
  const eigenvectors_sorted: number[][] = Array.from(
    { length: n },
    () => new Array(m),
  );

  for (let new_col = 0; new_col < m; new_col++) {
    const src_col = indexed[new_col].idx;

    let max_abs = 0;
    let max_row = 0;
    for (let row = 0; row < n; row++) {
      const abs_val = Math.abs(eigenvectors[row][src_col]);
      if (abs_val > max_abs) {
        max_abs = abs_val;
        max_row = row;
      }
    }

    const sign = eigenvectors[max_row][src_col] < 0 ? -1 : 1;

    for (let row = 0; row < n; row++) {
      eigenvectors_sorted[row][new_col] = sign * eigenvectors[row][src_col];
    }
  }

  return {
    eigenvalues: eigenvalues_sorted,
    eigenvectors: eigenvectors_sorted,
  };
}
