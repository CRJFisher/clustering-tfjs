import * as tf from '../backend/adapter';
import { deterministic_eigenpair_processing } from './post';
import { lanczos_smallest_eigenpairs, LanczosOperator } from './lanczos';
import { improved_jacobi_eigen } from './improved';

/**
 * Size threshold for choosing Lanczos over Jacobi.
 * For n <= threshold, Jacobi is used (proven accurate, fast enough).
 * For n > threshold, Lanczos is used (O(n²·m) vs O(n³)).
 */
const LANCZOS_THRESHOLD = 100;

/**
 * Eigenvalues at or below this value are treated as numerically zero when
 * counting near-zero eigenvalues (connected-component detection). Used by
 * both the Lanczos and Jacobi paths so that path routing at n=100 cannot
 * change the returned embedding dimension.
 *
 * Value is chosen above the Lanczos convergence tolerance (1e-6) so that
 * true structural zeros — which for a disconnected Laplacian can emerge
 * as small positives up to ~1e-6 before PSD clamping — are reliably
 * counted on both paths. Matches PSD_CLAMP_TOL in lanczos.ts.
 */
const NEAR_ZERO_TOL = 1e-5;

/**
 * Returns the `k` smallest eigenvectors AND eigenvalues of the provided symmetric matrix.
 * This is needed for spectral embedding normalization (dividing by D^{1/2}).
 *
 * Automatically selects the best eigensolver:
 * - n <= 100: Jacobi (full decomposition, proven accurate)
 * - n > 100: Lanczos (iterative, O(n²·m) vs O(n³))
 */
export function smallest_eigenvectors_with_values(
  matrix: tf.Tensor2D | LanczosOperator,
  k: number,
): { eigenvectors: tf.Tensor2D; eigenvalues: tf.Tensor1D } {
  if (!Number.isInteger(k) || k < 1) {
    throw new Error('k must be a positive integer.');
  }

  const is_operator = is_lanczos_operator(matrix);
  const n = is_operator ? matrix.n : matrix.shape[0];

  if (is_operator) {
    return lanczos_path(matrix, k, n, false);
  }

  if (n > LANCZOS_THRESHOLD && k < n / 3) {
    return lanczos_path(matrix, k, n, true);
  }

  return jacobi_path(matrix, k);
}

function is_lanczos_operator(
  matrix: tf.Tensor2D | LanczosOperator,
): matrix is LanczosOperator {
  return (
    typeof (matrix as LanczosOperator).n === 'number' &&
    typeof (matrix as LanczosOperator).matvec === 'function'
  );
}

function count_near_zeros(eigenvalues: number[]): number {
  let c = 0;
  for (const v of eigenvalues) {
    if (v <= NEAR_ZERO_TOL) c += 1;
    else break;
  }
  return c;
}

/**
 * Lanczos path: iterative eigensolver for large matrices.
 * Falls back to Jacobi if Lanczos fails.
 */
function lanczos_path(
  matrix: tf.Tensor2D | LanczosOperator,
  k: number,
  n: number,
  allow_jacobi_fallback: boolean,
): { eigenvectors: tf.Tensor2D; eigenvalues: tf.Tensor1D } {
  // Request extra eigenpairs to capture near-zero eigenvalues for component detection
  const k_request = Math.min(k + 5, n);

  try {
    let result = lanczos_smallest_eigenpairs(matrix, k_request, {
      is_psd: true,
      random_seed: 42,
    });

    let c = count_near_zeros(result.eigenvalues);

    // If near-zeros consumed the entire buffer, there may be more beyond k_request.
    // Retry with enough room to capture all of them.
    if (k + c > result.eigenvalues.length && result.eigenvalues.length < n) {
      const k_retry = Math.min(k + c + 5, n);
      result = lanczos_smallest_eigenpairs(matrix, k_retry, {
        is_psd: true,
        random_seed: 42,
      });
      c = count_near_zeros(result.eigenvalues);
    }

    const slice_cols = Math.min(k + c, result.eigenvalues.length);

    // Extract selected eigenpairs
    const selected_vecs: number[][] = Array.from(
      { length: n },
      () => new Array(slice_cols),
    );
    const selected_vals: number[] = new Array(slice_cols);

    for (let col = 0; col < slice_cols; col++) {
      selected_vals[col] = result.eigenvalues[col];
      for (let row = 0; row < n; row++) {
        selected_vecs[row][col] = result.eigenvectors[row][col];
      }
    }

    return {
      eigenvectors: tf.tensor2d(selected_vecs, [n, slice_cols], 'float32'),
      eigenvalues: tf.tensor1d(selected_vals, 'float32'),
    };
  } catch (err) {
    if (!allow_jacobi_fallback || is_lanczos_operator(matrix)) {
      throw err;
    }

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

    // Full eigendecomposition
    const { eigenvalues, eigenvectors } = improved_jacobi_eigen(matrix, {
      is_psd: true,
      max_iterations: 3000,
      tolerance: 1e-14,
    });

    // Deterministic ordering & sign fixing
    const processed = deterministic_eigenpair_processing({
      eigenvalues,
      eigenvectors,
    });

    const c = count_near_zeros(processed.eigenvalues);
    const n = processed.eigenvectors.length;
    const slice_cols = Math.min(k + c, n);

    // Extract selected eigenvectors
    const selected_vecs: number[][] = Array.from(
      { length: n },
      () => new Array(slice_cols),
    );
    const selected_vals: number[] = new Array(slice_cols);

    for (let col = 0; col < slice_cols; col++) {
      selected_vals[col] = processed.eigenvalues[col];
      for (let row = 0; row < n; row++) {
        selected_vecs[row][col] = processed.eigenvectors[row][col];
      }
    }

    return {
      eigenvectors: tf.tensor2d(selected_vecs, [n, slice_cols], 'float32'),
      eigenvalues: tf.tensor1d(selected_vals, 'float32'),
    };
  });
}
