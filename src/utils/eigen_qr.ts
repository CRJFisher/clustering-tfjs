import * as tf from '../tf-adapter';

/**
 * QR Algorithm-based eigendecomposition for symmetric matrices.
 *
 * The QR algorithm is more numerically stable than Jacobi iteration
 * and converges faster for most matrices. This implementation uses
 * TensorFlow.js's built-in QR decomposition.
 *
 * Algorithm:
 * 1. Start with A₀ = A
 * 2. For each iteration:
 *    - Compute QR decomposition: Aᵢ = QᵢRᵢ
 *    - Form Aᵢ₊₁ = RᵢQᵢ
 * 3. As i → ∞, Aᵢ converges to a diagonal matrix of eigenvalues
 * 4. The product Q₀Q₁...Qᵢ gives the eigenvectors
 */
export function qr_eigen_decomposition(
  matrix: tf.Tensor2D,
  {
    maxIterations = 1000,
    tolerance = 1e-10,
  }: { maxIterations?: number; tolerance?: number } = {},
): { eigenvalues: number[]; eigenvectors: number[][] } {
  return tf.tidy(() => {
    const n = matrix.shape[0];

    // Initialize
    let A = matrix.clone();
    let V = tf.eye(n); // Accumulate eigenvector transformations

    // Helper to compute off-diagonal norm
    const offDiagonalNorm = (M: tf.Tensor2D): number => {
      const data = M.arraySync() as number[][];
      let sum = 0;
      for (let i = 0; i < n; i++) {
        for (let j = 0; j < n; j++) {
          if (i !== j) {
            sum += data[i][j] * data[i][j];
          }
        }
      }
      return Math.sqrt(sum);
    };

    let iter = 0;
    let offDiag = offDiagonalNorm(A as tf.Tensor2D);

    // Apply Wilkinson shift for better convergence on small eigenvalues
    const wilkinsonShift = (M: tf.Tensor2D): number => {
      if (n < 2) return 0;

      const data = M.arraySync() as number[][];
      const a = data[n - 2][n - 2];
      const b = data[n - 2][n - 1];
      const c = data[n - 1][n - 1];

      // Off-diagonal is zero: submatrix is already diagonal, shift to c
      if (b === 0) return c;

      // Compute shift as eigenvalue of 2x2 bottom-right submatrix closest to c
      const delta = (a - c) / 2;
      const sign = delta >= 0 ? 1 : -1;
      return (
        c -
        (sign * b * b) / (Math.abs(delta) + Math.sqrt(delta * delta + b * b))
      );
    };

    while (iter < maxIterations && offDiag > tolerance) {
      // Apply shift for better convergence
      const shift = wilkinsonShift(A);
      const I = tf.eye(n);
      const A_shifted: tf.Tensor2D = shift !== 0
        ? A.sub(I.mul(shift)) as tf.Tensor2D
        : A;

      // QR decomposition
      const [Q, R] = tf.linalg.qr(A_shifted);

      // When the Wilkinson shift equals an exact eigenvalue, A_shifted is
      // singular and tf.linalg.qr can produce NaN on some platforms.
      // Detect this and retry without the shift. Float32 precision makes
      // small perturbations invisible, so falling back to unshifted QR
      // (which always works on a valid symmetric matrix) is the safest option.
      const R_data = R.arraySync() as number[][];
      const qrHasNaN = R_data.some(row => row.some(v => !isFinite(v)));

      if (qrHasNaN) {
        Q.dispose();
        R.dispose();
        if (shift !== 0) {
          I.dispose();
          A_shifted.dispose();
        }

        // Unshifted QR step — always valid for non-singular input
        const [Q2, R2] = tf.linalg.qr(A);
        const A_new = R2.matMul(Q2) as tf.Tensor2D;
        const V_new = V.matMul(Q2) as tf.Tensor2D;

        A.dispose();
        V.dispose();
        A = A_new;
        V = V_new;

        offDiag = offDiagonalNorm(A);
        Q2.dispose();
        R2.dispose();
        iter++;
        continue;
      }

      // Update A = RQ + shift*I
      A.dispose();
      A = R.matMul(Q) as tf.Tensor2D;
      if (shift !== 0) {
        const temp = A;
        A = A.add(I.mul(shift)) as tf.Tensor2D;
        temp.dispose();
      }

      // Accumulate eigenvector transformations
      const V_new = V.matMul(Q) as tf.Tensor2D;
      V.dispose();
      V = V_new;

      // Check convergence
      offDiag = offDiagonalNorm(A);

      // Cleanup
      Q.dispose();
      R.dispose();
      I.dispose();
      if (shift !== 0) A_shifted.dispose();

      iter++;
    }

    if (iter === maxIterations) {
      console.warn(
        `QR algorithm did not converge after ${maxIterations} iterations. Final off-diagonal norm: ${offDiag}`,
      );
    }

    // Extract eigenvalues from diagonal
    const A_data = A.arraySync() as number[][];
    const eigenvalues = A_data.map((row, i) => row[i]);

    // Extract eigenvectors
    const V_data = V.arraySync() as number[][];

    // Sort by eigenvalue (ascending)
    const indexed = eigenvalues.map((val, idx) => ({ val, idx }));
    indexed.sort((a, b) => a.val - b.val);

    const sortedValues = indexed.map((p) => p.val);
    const sortedVectors: number[][] = Array(n)
      .fill(0)
      .map(() => Array(n).fill(0));

    // Rearrange eigenvector columns
    for (let newIdx = 0; newIdx < n; newIdx++) {
      const oldIdx = indexed[newIdx].idx;
      for (let row = 0; row < n; row++) {
        sortedVectors[row][newIdx] = V_data[row][oldIdx];
      }
    }

    return { eigenvalues: sortedValues, eigenvectors: sortedVectors };
  });
}

/**
 * Specialized QR algorithm for tridiagonal matrices.
 * Since normalized Laplacians are often nearly tridiagonal after
 * similarity transformations, this can be more efficient.
 */
export function tridiagonal_qr_eigen(
  diagonal: number[],
  offDiagonal: number[],
  computeVectors: boolean = true,
): { eigenvalues: number[]; eigenvectors?: number[][] } {
  const n = diagonal.length;

  // Clone arrays to avoid mutation
  const d = [...diagonal];
  const e = [...offDiagonal, 0]; // Pad with 0 for convenience

  let V: number[][] | undefined;
  if (computeVectors) {
    V = Array(n)
      .fill(0)
      .map((_, i) =>
        Array(n)
          .fill(0)
          .map((_, j) => (i === j ? 1 : 0)),
      );
  }

  // QL algorithm for tridiagonal matrices (Numerical Recipes tqli)
  for (let i = 0; i < n - 1; i++) {
    let iter = 0;
    let m: number;

    do {
      // Find small off-diagonal element
      for (m = i; m < n - 1; m++) {
        const dd = Math.abs(d[m]) + Math.abs(d[m + 1]);
        if (Math.abs(e[m]) <= Number.EPSILON * dd) break;
      }

      if (m !== i) {
        if (iter++ === 30) {
          console.warn('Tridiagonal QR: Too many iterations');
          break;
        }

        // Compute shift using implicit QL formulation
        let g = (d[i + 1] - d[i]) / (2 * e[i]);
        let r = Math.sqrt(g * g + 1);
        g = d[m] - d[i] + e[i] / (g + (g >= 0 ? r : -r));

        let s = 1,
          c = 1,
          p = 0;

        for (let j = m - 1; j >= i; j--) {
          const f = s * e[j];
          const b = c * e[j];
          r = Math.sqrt(f * f + g * g);
          e[j + 1] = r;

          if (r === 0) {
            d[j + 1] -= p;
            e[m] = 0;
            break;
          }

          s = f / r;
          c = g / r;
          g = d[j + 1] - p;
          r = (d[j] - g) * s + 2 * c * b;
          p = s * r;
          d[j + 1] = g + p;
          g = c * r - b;

          // Update eigenvectors
          if (computeVectors && V) {
            for (let k = 0; k < n; k++) {
              const vt = V[k][j + 1];
              V[k][j + 1] = s * V[k][j] + c * vt;
              V[k][j] = c * V[k][j] - s * vt;
            }
          }
        }

        d[i] -= p;
        e[i] = g;
        e[m] = 0;
      }
    } while (m !== i);
  }

  // Sort eigenvalues and eigenvectors
  const indexed = d.map((val, idx) => ({ val, idx }));
  indexed.sort((a, b) => a.val - b.val);

  const eigenvalues = indexed.map((p) => p.val);

  let eigenvectors: number[][] | undefined;
  if (computeVectors && V) {
    eigenvectors = Array(n)
      .fill(0)
      .map(() => Array(n).fill(0));
    for (let newIdx = 0; newIdx < n; newIdx++) {
      const oldIdx = indexed[newIdx].idx;
      for (let row = 0; row < n; row++) {
        eigenvectors[row][newIdx] = V[row][oldIdx];
      }
    }
  }

  return { eigenvalues, eigenvectors };
}
