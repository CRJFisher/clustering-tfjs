import tf from '../tf-adapter';

/**
 * Improved Jacobi eigendecomposition for symmetric matrices.
 *
 * Enhancements over the basic Jacobi method:
 * 1. Cyclic Jacobi - systematically sweep through all pairs
 * 2. Threshold scaling - adapt threshold as we converge
 * 3. Better handling of small pivots
 * 4. Post-processing to ensure non-negative eigenvalues for PSD matrices
 */
export function improved_jacobi_eigen(
  matrix: tf.Tensor2D,
  {
    maxIterations = 3000,
    tolerance = 1e-14,
    isPSD = false, // Is matrix positive semi-definite?
  }: { maxIterations?: number; tolerance?: number; isPSD?: boolean } = {},
): { eigenvalues: number[]; eigenvectors: number[][] } {
  if (matrix.shape.length !== 2 || matrix.shape[0] !== matrix.shape[1]) {
    throw new Error('Input tensor must be square (n × n).');
  }

  const A = matrix.arraySync() as number[][];
  const n = A.length;

  // Check if already diagonal
  const isApproximatelyDiagonal = (): boolean => {
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        if (i === j) continue;
        if (Math.abs(A[i][j]) > tolerance * 10) return false;
      }
    }
    return true;
  };

  if (isApproximatelyDiagonal()) {
    const diag = A.map((row, i) => row[i]);
    const V = A.map((_, i) =>
      Array.from({ length: n }, (_, j) => (i === j ? 1 : 0)),
    );
    return { eigenvalues: diag, eigenvectors: V };
  }

  // Deep copy for working matrix
  const D: number[][] = A.map((row) => [...row]);

  // Eigenvector accumulator
  const V: number[][] = Array.from({ length: n }, (_, i) =>
    Array.from({ length: n }, (_, j) => (i === j ? 1 : 0)),
  );

  // Compute Frobenius norm of off-diagonal elements
  const offDiagNorm = (M: number[][]): number => {
    let sum = 0;
    for (let i = 0; i < n; i++) {
      for (let j = i + 1; j < n; j++) {
        sum += M[i][j] * M[i][j];
      }
    }
    return Math.sqrt(2 * sum); // Account for symmetry
  };

  let sweep = 0;
  let changed = true;

  // Adaptive threshold based on matrix norm
  const matrixNorm = Math.sqrt(
    A.reduce((sum, row) => sum + row.reduce((s, v) => s + v * v, 0), 0),
  );
  let threshold = tolerance * matrixNorm;

  while (sweep < maxIterations && changed) {
    changed = false;

    // Cyclic Jacobi: sweep through all pairs
    for (let p = 0; p < n - 1; p++) {
      for (let q = p + 1; q < n; q++) {
        const a_pq = D[p][q];

        // Skip if already small
        if (Math.abs(a_pq) < threshold) continue;

        changed = true;
        const a_pp = D[p][p];
        const a_qq = D[q][q];

        // Compute rotation angle
        const diff = a_qq - a_pp;
        let t: number;

        if (Math.abs(a_pq) < Math.abs(diff) * 1e-15) {
          // Very small angle - use approximation
          t = a_pq / diff;
        } else {
          const theta = diff / (2 * a_pq);
          t = 1 / (Math.abs(theta) + Math.sqrt(1 + theta * theta));
          if (theta < 0) t = -t;
        }

        const c = 1 / Math.sqrt(1 + t * t);
        const s = t * c;
        const tau = s / (1 + c);

        // Update diagonal elements
        D[p][p] = a_pp - t * a_pq;
        D[q][q] = a_qq + t * a_pq;
        D[p][q] = D[q][p] = 0;

        // Update row p and column p (j < p)
        for (let j = 0; j < p; j++) {
          const g = D[j][p];
          const h = D[j][q];
          D[j][p] = D[p][j] = g - s * (h + g * tau);
          D[j][q] = D[q][j] = h + s * (g - h * tau);
        }

        // Update row p and column p (p < j < q)
        for (let j = p + 1; j < q; j++) {
          const g = D[p][j];
          const h = D[j][q];
          D[p][j] = D[j][p] = g - s * (h + g * tau);
          D[j][q] = D[q][j] = h + s * (g - h * tau);
        }

        // Update row p and column p (j > q)
        for (let j = q + 1; j < n; j++) {
          const g = D[p][j];
          const h = D[q][j];
          D[p][j] = D[j][p] = g - s * (h + g * tau);
          D[q][j] = D[j][q] = h + s * (g - h * tau);
        }

        // Update eigenvectors
        for (let j = 0; j < n; j++) {
          const g = V[j][p];
          const h = V[j][q];
          V[j][p] = g - s * (h + g * tau);
          V[j][q] = h + s * (g - h * tau);
        }
      }
    }

    sweep++;

    // Adaptive threshold reduction
    if (sweep % 5 === 0) {
      const currentNorm = offDiagNorm(D);
      if (currentNorm < threshold * n) {
        threshold *= 0.1;
      }
    }
  }

  if (sweep === maxIterations) {
    console.warn(
      `Improved Jacobi solver reached max iterations (${maxIterations}). ` +
        `Final off-diagonal norm: ${offDiagNorm(D).toExponential(3)}`,
    );
  }

  // Extract eigenvalues
  let eigenvalues: number[] = D.map((row, i) => row[i]);

  // Post-processing for PSD matrices
  if (isPSD) {
    // Clamp small negative values to zero
    // For normalized Laplacians, eigenvalues should be in [0, 2]
    // Any negative value is due to numerical error
    const threshold = 1e-8; // More relaxed threshold for PSD matrices
    eigenvalues = eigenvalues.map((v) => (v < threshold ? 0 : v));
  }

  // Sort eigen-pairs
  const indexed = eigenvalues.map((val, idx) => ({ val, idx }));
  indexed.sort((a, b) => a.val - b.val);

  const sortedValues = indexed.map((p) => p.val);
  const sortedVectors: number[][] = Array.from(
    { length: n },
    () => new Array(n),
  );

  for (let newIdx = 0; newIdx < n; newIdx++) {
    const oldIdx = indexed[newIdx].idx;
    for (let row = 0; row < n; row++) {
      sortedVectors[row][newIdx] = V[row][oldIdx];
    }
  }

  return { eigenvalues: sortedValues, eigenvectors: sortedVectors };
}

/**
 * Specialized version for normalized Laplacians.
 * Takes advantage of the known properties:
 * - Symmetric
 * - Positive semi-definite
 * - Eigenvalues in [0, 2]
 * - Smallest eigenvalue(s) ≈ 0 for connected components
 */
export function laplacian_eigen_decomposition(
  laplacian: tf.Tensor2D,
  k: number,
): tf.Tensor2D {
  return tf.tidy(() => {
    const { eigenvalues, eigenvectors } = improved_jacobi_eigen(laplacian, {
      isPSD: true,
      maxIterations: 3000,
      tolerance: 1e-14,
    });

    // For Laplacians, we know smallest eigenvalues should be very close to 0
    // Count how many are numerically zero
    const TOL = 1e-5; // More relaxed than general case
    let numZeros = 0;
    for (const val of eigenvalues) {
      if (val <= TOL) numZeros++;
      else break;
    }

    // Return k + numZeros columns
    const n = eigenvectors.length;
    const numCols = Math.min(k + numZeros, n);

    const selected: number[][] = Array.from(
      { length: n },
      () => new Array(numCols),
    );

    for (let col = 0; col < numCols; col++) {
      for (let row = 0; row < n; row++) {
        selected[row][col] = eigenvectors[row][col];
      }
    }

    return tf.tensor2d(selected, [n, numCols], 'float32');
  });
}
