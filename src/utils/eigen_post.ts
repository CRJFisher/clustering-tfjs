/**
 * Deterministic eigen-pair post–processing.
 *
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
  eigenvectors: number[][]; // shape (n, n) – column j belongs to eigenvalue j
}

export interface EigenPairOutput {
  eigenvalues: number[];
  eigenvectors: number[][]; // same shape, processed per rules above
}

/**
 * Applies deterministic ordering and sign convention to raw eigen-pairs.
 */
export function deterministic_eigenpair_processing(
  input: EigenPairInput,
): EigenPairOutput {
  const { eigenvalues, eigenvectors } = input;

  if (eigenvectors.length === 0) {
    return { eigenvalues: [], eigenvectors: [] };
  }

  const n = eigenvectors.length;

  // Validate shape (n rows, n columns)
  if (eigenvectors.some((row) => row.length !== n) || eigenvalues.length !== n) {
    throw new Error(
      "eigenvectors must be square (n×n) and eigenvalues length must equal n.",
    );
  }

  // Step 1: sort indices by ascending eigenvalue
  const indexed = eigenvalues.map((val, idx) => ({ val, idx }));
  indexed.sort((a, b) => a.val - b.val);

  const eigenvaluesSorted: number[] = indexed.map((p) => p.val);
  const eigenvectorsSorted: number[][] = Array.from({ length: n }, () => new Array(n));

  // Step 2: for each eigenvector apply sign fix while copying into new matrix
  for (let newCol = 0; newCol < n; newCol++) {
    const srcCol = indexed[newCol].idx;

    // Find entry with maximum absolute magnitude
    let maxAbs = 0;
    let maxRow = 0;
    for (let row = 0; row < n; row++) {
      const absVal = Math.abs(eigenvectors[row][srcCol]);
      if (absVal > maxAbs) {
        maxAbs = absVal;
        maxRow = row;
      }
    }

    const sign = eigenvectors[maxRow][srcCol] < 0 ? -1 : 1;

    for (let row = 0; row < n; row++) {
      eigenvectorsSorted[row][newCol] = sign * eigenvectors[row][srcCol];
    }
  }

  return { eigenvalues: eigenvaluesSorted, eigenvectors: eigenvectorsSorted };
}
