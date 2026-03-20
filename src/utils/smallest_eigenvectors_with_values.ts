import * as tf from '../tf-adapter';
import { deterministic_eigenpair_processing } from './eigen_post';

/**
 * Size threshold for choosing Lanczos over Jacobi.
 * For n <= threshold, Jacobi is used (proven accurate, fast enough).
 * For n > threshold, Lanczos is used (O(n²·m) vs O(n³)).
 */
const LANCZOS_THRESHOLD = 100;

/**
 * Returns the `k` smallest eigenvectors AND eigenvalues of the provided symmetric matrix.
 * This is needed for spectral embedding normalization (dividing by D^{1/2}).
 *
 * Automatically selects the best eigensolver:
 * - n <= 100: Jacobi (full decomposition, proven accurate)
 * - n > 100: Lanczos (iterative, O(n²·m) vs O(n³))
 */
export function smallest_eigenvectors_with_values(
  matrix: tf.Tensor2D,
  k: number,
): { eigenvectors: tf.Tensor2D; eigenvalues: tf.Tensor1D } {
  if (!Number.isInteger(k) || k < 1) {
    throw new Error('k must be a positive integer.');
  }

  const n = matrix.shape[0];

  if (n > LANCZOS_THRESHOLD && k < n / 3) {
    return lanczos_path(matrix, k, n);
  }

  return jacobi_path(matrix, k);
}

/**
 * Lanczos path: iterative eigensolver for large matrices.
 * Falls back to Jacobi if Lanczos fails.
 */
function lanczos_path(
  matrix: tf.Tensor2D,
  k: number,
  n: number,
): { eigenvectors: tf.Tensor2D; eigenvalues: tf.Tensor1D } {
  // Request extra eigenpairs to capture near-zero eigenvalues for component detection
  const kRequest = Math.min(k + 5, n);

  try {
    // eslint-disable-next-line @typescript-eslint/no-var-requires
    const { lanczos_smallest_eigenpairs } = require('./lanczos');
    const result = lanczos_smallest_eigenpairs(matrix, kRequest, {
      isPSD: true,
      randomSeed: 42,
    });

    // Determine number of numerically-zero eigenvalues
    const TOL = 1e-2;
    let c = 0;
    for (const v of result.eigenvalues) {
      if (v <= TOL) c += 1;
      else break;
    }

    const sliceCols = Math.min(k + c, result.eigenvalues.length);

    // Extract selected eigenpairs
    const selectedVecs: number[][] = Array.from(
      { length: n },
      () => new Array(sliceCols),
    );
    const selectedVals: number[] = new Array(sliceCols);

    for (let col = 0; col < sliceCols; col++) {
      selectedVals[col] = result.eigenvalues[col];
      for (let row = 0; row < n; row++) {
        selectedVecs[row][col] = result.eigenvectors[row][col];
      }
    }

    return {
      eigenvectors: tf.tensor2d(selectedVecs, [n, sliceCols], 'float32'),
      eigenvalues: tf.tensor1d(selectedVals, 'float32'),
    };
  } catch (err) {
    console.warn(
      `[spectral] Lanczos solver failed, falling back to Jacobi: ${err instanceof Error ? err.message : String(err)}`,
    );
    return jacobi_path(matrix, k);
  }
}

/**
 * Jacobi path: full eigendecomposition for small matrices or as fallback.
 */
function jacobi_path(
  matrix: tf.Tensor2D,
  k: number,
): { eigenvectors: tf.Tensor2D; eigenvalues: tf.Tensor1D } {
  return tf.tidy(() => {
    // eslint-disable-next-line @typescript-eslint/no-var-requires
    const { improved_jacobi_eigen } = require('./eigen_improved');

    // Full eigendecomposition
    const { eigenvalues, eigenvectors } = improved_jacobi_eigen(matrix, {
      isPSD: true,
      maxIterations: 3000,
      tolerance: 1e-14,
    });

    // Deterministic ordering & sign fixing
    const processed = deterministic_eigenpair_processing({
      eigenvalues,
      eigenvectors,
    });

    // Determine number of numerically-zero eigenvalues
    const TOL = 1e-7;
    let c = 0;
    for (const v of processed.eigenvalues) {
      if (v <= TOL) c += 1;
      else break;
    }

    const n = processed.eigenvectors.length;
    const sliceCols = Math.min(k + c, n);

    // Extract selected eigenvectors
    const selectedVecs: number[][] = Array.from(
      { length: n },
      () => new Array(sliceCols),
    );
    const selectedVals: number[] = new Array(sliceCols);

    for (let col = 0; col < sliceCols; col++) {
      selectedVals[col] = processed.eigenvalues[col];
      for (let row = 0; row < n; row++) {
        selectedVecs[row][col] = processed.eigenvectors[row][col];
      }
    }

    return {
      eigenvectors: tf.tensor2d(selectedVecs, [n, sliceCols], 'float32'),
      eigenvalues: tf.tensor1d(selectedVals, 'float32'),
    };
  });
}
