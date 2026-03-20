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
  eigenvectors: number[][]; // shape (n, m) – column j belongs to eigenvalue j (m may equal n or k)
}

export interface EigenPairOutput {
  /**
   * Eigen-values sorted in ascending order.  We expose them under two
   * property names to stay compatible with the acceptance criteria drafted
   * in task-12.3.1 (valuesSorted) *and* with existing internal call-sites
   * (eigenvalues).
   */
  eigenvalues: number[];
  /** Alias – kept for backwards-compatibility with task spec */
  valuesSorted: number[];

  /**
   * Column-wise eigen-vectors after sign correction.
   * Same dual naming scheme as for the eigen-values.
   */
  eigenvectors: number[][]; // shape (n, m)
  /** Alias matching task spec wording */
  vectorsSorted: number[][];
}

/**
 * Applies deterministic ordering and sign convention to raw eigen-pairs.
 */
export function deterministic_eigenpair_processing(
  input: EigenPairInput,
): EigenPairOutput {
  const { eigenvalues, eigenvectors } = input;

  if (eigenvectors.length === 0) {
    return {
      eigenvalues: [],
      valuesSorted: [],
      eigenvectors: [],
      vectorsSorted: [],
    };
  }

  const n = eigenvectors.length;       // number of rows (samples)
  const m = eigenvalues.length;        // number of eigenpairs (may be < n)

  // Validate shape: each row must have m columns, matching eigenvalues count
  if (eigenvectors.some((row) => row.length !== m)) {
    throw new Error(
      `eigenvectors column count must match eigenvalues length (${m}).`,
    );
  }

  // Step 1: sort indices by ascending eigenvalue
  const indexed = eigenvalues.map((val, idx) => ({ val, idx }));
  indexed.sort((a, b) => a.val - b.val);

  const eigenvaluesSorted: number[] = indexed.map((p) => p.val);
  const eigenvectorsSorted: number[][] = Array.from(
    { length: n },
    () => new Array(m),
  );

  // Step 2: for each eigenvector apply sign fix while copying into new matrix
  for (let newCol = 0; newCol < m; newCol++) {
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

  return {
    eigenvalues: eigenvaluesSorted,
    valuesSorted: eigenvaluesSorted,
    eigenvectors: eigenvectorsSorted,
    vectorsSorted: eigenvectorsSorted,
  };
}
