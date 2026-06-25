import * as tf from '../backend/adapter';
import { reorthogonalize_vector } from './orthogonalize';
import { make_random_stream } from '../random';

export interface LanczosOptions {
  /** Maximum Lanczos subspace size before restart. Default: min(max(2k+20, 4k), n, 200) */
  max_subspace_size?: number;
  max_restarts?: number;
  convergence_tol?: number;
  /** Random seed for deterministic starting vector. Default: 42 */
  random_seed?: number;
  /** Whether the matrix is PSD (clamp negative eigenvalues to 0). Default: true */
  is_psd?: boolean;
}

export interface LanczosResult {
  eigenvalues: number[];
  eigenvectors: number[][];
}

export interface LanczosOperator {
  n: number;
  matvec: (vector: Float64Array) => Float64Array;
}

const CONVERGENCE_TOL_DEFAULT = 1e-6;
const BREAKDOWN_TOL = 1e-10;
const PSD_CLAMP_TOL = 1e-5;
const NORM_REDUCTION_TRIGGER = 0.7071; // 1/sqrt(2) — triggers second reorth pass

/**
 * Computes the k smallest eigenpairs of a symmetric matrix using the
 * Lanczos algorithm with full reorthogonalization.
 *
 * Standard Lanczos: applies A*v directly and extracts the k smallest
 * Ritz values from the tridiagonal eigenproblem. Uses simple restart
 * (best Ritz vector as new starting point) when the subspace reaches
 * its maximum size without convergence.
 *
 * Complexity: O(n² · m) where m is the Lanczos subspace size (typically 30–100),
 * versus O(n³) for Jacobi. For n=5000, k=5, this is ~1000x faster.
 *
 * @param matrix  Symmetric n×n matrix (typically a normalized Laplacian)
 */
export function lanczos_smallest_eigenpairs(
  matrix: tf.Tensor2D | LanczosOperator,
  k: number,
  options?: LanczosOptions,
): LanczosResult {
  const is_operator = is_lanczos_operator(matrix);
  const n = is_operator ? matrix.n : matrix.shape[0];

  if (!Number.isInteger(k) || k < 1) {
    throw new Error('k must be a positive integer >= 1.');
  }
  if (k > n) {
    throw new Error(`k (${k}) cannot exceed matrix size n (${n}).`);
  }
  if (!is_operator && matrix.shape[0] !== matrix.shape[1]) {
    throw new Error('Input matrix must be square.');
  }

  const {
    max_subspace_size: max_subspace_size_opt,
    max_restarts = 10,
    convergence_tol = CONVERGENCE_TOL_DEFAULT,
    random_seed = 42,
    is_psd = true,
  } = options ?? {};

  // Subspace size: need enough room beyond k for convergence
  const m_max = max_subspace_size_opt ?? Math.min(Math.max(2 * k + 20, 4 * k), n, 200);

  // Dense tensors are extracted once to avoid repeated GPU→CPU transfers.
  // Matrix-free sparse callers provide their own matvec and never densify.
  const A = is_operator ? null : matrix.arraySync() as number[][];
  const apply_matvec = (v: Float64Array): Float64Array =>
    is_operator ? matrix.matvec(v) : dense_matvec(A!, v, n);

  // Lanczos converges to extreme eigenvalues (both largest and smallest).
  // We extract the k smallest Ritz values from the tridiagonal eigenproblem.

  const rng = make_random_stream(random_seed);

  const q = new Float64Array(n);
  let norm = 0;
  for (let i = 0; i < n; i++) {
    q[i] = rng.rand() - 0.5;
    norm += q[i] * q[i];
  }
  norm = Math.sqrt(norm);
  for (let i = 0; i < n; i++) q[i] /= norm;

  let basis_vectors: Float64Array[] = [q];
  let alpha: number[] = [];
  let beta: number[] = [];

  let converged_eigenvalues: number[] | null = null;
  let converged_eigenvectors: number[][] | null = null;

  for (let restart = 0; restart <= max_restarts; restart++) {
    const start_j = alpha.length; // resumption point after restart

    for (let j = start_j; j < m_max; j++) {
      const v_j = basis_vectors[j];

      // w = A * v_j
      const w = apply_matvec(v_j);

      // w = w - beta_{j-1} * v_{j-1}
      if (j > 0) {
        const beta_prev = beta[j - 1];
        const v_prev = basis_vectors[j - 1];
        for (let i = 0; i < n; i++) w[i] -= beta_prev * v_prev[i];
      }

      // alpha_j = v_j^T * w
      let alpha_j = 0;
      for (let i = 0; i < n; i++) alpha_j += v_j[i] * w[i];
      alpha.push(alpha_j);

      // w = w - alpha_j * v_j
      for (let i = 0; i < n; i++) w[i] -= alpha_j * v_j[i];

      // Full reorthogonalization against all previous basis vectors
      const norm_before = vec_norm(w, n);
      reorthogonalize(w, basis_vectors, n);
      const norm_after = vec_norm(w, n);

      // Double Gram-Schmidt if significant cancellation detected
      if (norm_after < NORM_REDUCTION_TRIGGER * norm_before) {
        reorthogonalize(w, basis_vectors, n);
      }

      const beta_j = vec_norm(w, n);

      // Check for lucky breakdown (invariant subspace found)
      if (beta_j < BREAKDOWN_TOL) {
        // We've found an invariant subspace. Solve for what we have.
        const { values, vectors } = tridiagonal_ql(
          alpha.slice(),
          beta.slice(0, alpha.length - 1),
        );

        const result = extract_smallest(
          values, vectors, basis_vectors, k, n, is_psd,
        );

        if (result.eigenvalues.length >= k) {
          return result;
        }

        // Not enough eigenpairs — generate random restart vector
        // orthogonal to current basis
        const new_q = random_orthogonal_vector(basis_vectors, n, rng);
        if (new_q === null) {
          // Entire space spanned — return what we have
          return result;
        }
        beta.push(0);
        basis_vectors.push(new_q);
        continue;
      }

      beta.push(beta_j);

      // Normalize and append new basis vector
      const q_new = new Float64Array(n);
      for (let i = 0; i < n; i++) q_new[i] = w[i] / beta_j;
      basis_vectors.push(q_new);

      // Check convergence periodically (every 5 iterations after minimum)
      if (j >= 2 * k && (j - start_j) % 5 === 4) {
        const { values, vectors } = tridiagonal_ql(
          alpha.slice(),
          beta.slice(0, alpha.length - 1),
        );

        const converged = check_convergence(
          values, vectors, beta[beta.length - 1],
          k, convergence_tol,
        );

        if (converged) {
          return extract_smallest(
            values, vectors, basis_vectors, k, n, is_psd,
          );
        }
      }
    }

    // Reached max subspace size — perform simple restart
    const { values, vectors } = tridiagonal_ql(
      alpha.slice(),
      beta.slice(0, alpha.length - 1),
    );

    const converged = check_convergence(
      values, vectors, beta[beta.length - 1],
      k, convergence_tol,
    );

    if (converged) {
      return extract_smallest(
        values, vectors, basis_vectors, k, n, is_psd,
      );
    }

    // Simple restart: use the best Ritz vector as new starting vector
    const restart_result = extract_smallest(
      values, vectors, basis_vectors, k, n, is_psd,
    );
    converged_eigenvalues = restart_result.eigenvalues;
    converged_eigenvectors = restart_result.eigenvectors;

    const m = alpha.length;
    const indices = values.map((v, i) => ({ v, i }));
    indices.sort((a, b) => a.v - b.v);

    const best_idx = indices[0].i;
    const restart_vec = new Float64Array(n);
    for (let row = 0; row < n; row++) {
      let sum = 0;
      for (let col = 0; col < m; col++) {
        sum += basis_vectors[col][row] * vectors[col][best_idx];
      }
      restart_vec[row] = sum;
    }
    const restart_norm = vec_norm(restart_vec, n);
    if (restart_norm > BREAKDOWN_TOL) {
      for (let i = 0; i < n; i++) restart_vec[i] /= restart_norm;
    }

    basis_vectors = [restart_vec];
    alpha = [];
    beta = [];
  }

  // Max restarts exhausted — return best result so far
  if (converged_eigenvalues !== null && converged_eigenvectors !== null) {
    console.warn(
      `[lanczos] Did not fully converge after ${max_restarts} restarts. Returning best approximation.`,
    );
    return {
      eigenvalues: converged_eigenvalues,
      eigenvectors: converged_eigenvectors,
    };
  }

  // Final attempt: solve whatever tridiagonal we have
  const { values, vectors } = tridiagonal_ql(alpha.slice(), beta.slice(0, alpha.length - 1));
  return extract_smallest(values, vectors, basis_vectors, k, n, is_psd);
}


function is_lanczos_operator(
  matrix: tf.Tensor2D | LanczosOperator,
): matrix is LanczosOperator {
  return (
    typeof (matrix as LanczosOperator).n === 'number' &&
    typeof (matrix as LanczosOperator).matvec === 'function'
  );
}

function dense_matvec(
  A: number[][],
  v: Float64Array,
  n: number,
): Float64Array {
  const result = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    let sum = 0;
    const row = A[i];
    for (let j = 0; j < n; j++) {
      sum += row[j] * v[j];
    }
    result[i] = sum;
  }
  return result;
}

function vec_norm(v: Float64Array, n: number): number {
  let sum = 0;
  for (let i = 0; i < n; i++) sum += v[i] * v[i];
  return Math.sqrt(sum);
}

const reorthogonalize = reorthogonalize_vector<Float64Array>;

/**
 * Generates a random unit vector orthogonal to all basis vectors.
 * Returns null if the basis already spans the full space.
 */
function random_orthogonal_vector(
  basis: Float64Array[],
  n: number,
  rng: ReturnType<typeof make_random_stream>,
): Float64Array | null {
  for (let attempt = 0; attempt < 5; attempt++) {
    const v = new Float64Array(n);
    for (let i = 0; i < n; i++) v[i] = rng.rand() - 0.5;

    // Project out all basis components (twice for stability)
    reorthogonalize(v, basis, n);
    reorthogonalize(v, basis, n);

    const norm = vec_norm(v, n);
    if (norm > BREAKDOWN_TOL) {
      for (let i = 0; i < n; i++) v[i] /= norm;
      return v;
    }
  }
  return null;
}

/**
 * Check if the k smallest eigenvalues have converged.
 * Uses the residual bound: |beta_m * s_i_last| < tol * max(1, |theta_i|)
 */
function check_convergence(
  values: number[],
  vectors: number[][],
  last_beta: number,
  k: number,
  tol: number,
): boolean {
  const m = values.length;
  if (m < k) return false;

  const indices = values.map((v, i) => ({ v, i }));
  indices.sort((a, b) => a.v - b.v);

  for (let r = 0; r < k; r++) {
    const idx = indices[r].i;
    const last_component = Math.abs(vectors[m - 1][idx]);
    const residual = Math.abs(last_beta * last_component);
    const threshold = tol * Math.max(1, Math.abs(values[idx]));

    if (residual > threshold) {
      return false;
    }
  }

  return true;
}

function extract_smallest(
  ritz_values: number[],
  ritz_vectors: number[][],
  basis_vectors: Float64Array[],
  k: number,
  n: number,
  is_psd: boolean,
): LanczosResult {
  const m = ritz_values.length;

  const indices = ritz_values.map((v, i) => ({ v, i }));
  indices.sort((a, b) => a.v - b.v);

  const use_k = Math.min(k, m);
  const eigenvalues: number[] = [];
  const eigenvectors: number[][] = Array.from({ length: n }, () => new Array(use_k));

  for (let r = 0; r < use_k; r++) {
    const idx = indices[r].i;

    let lambda = ritz_values[idx];

    if (is_psd && lambda < 0) {
      if (lambda < -PSD_CLAMP_TOL) {
        console.warn(
          `[lanczos] Large negative eigenvalue ${lambda} clamped to 0 (exceeds tolerance ${PSD_CLAMP_TOL}).`,
        );
      }
      lambda = 0;
    }

    eigenvalues.push(lambda);

    // Reconstruct Ritz vector: u = Q * s_idx
    // Q is [n, m_basis] where m_basis = basis_vectors.length, s is [m, 1]
    const basis_count = Math.min(basis_vectors.length, m);
    for (let row = 0; row < n; row++) {
      let sum = 0;
      for (let col = 0; col < basis_count; col++) {
        sum += basis_vectors[col][row] * ritz_vectors[col][idx];
      }
      eigenvectors[row][r] = sum;
    }
  }

  // Apply deterministic sign convention: largest-magnitude component positive
  for (let col = 0; col < use_k; col++) {
    let max_abs = 0;
    let max_row = 0;
    for (let row = 0; row < n; row++) {
      const abs_val = Math.abs(eigenvectors[row][col]);
      if (abs_val > max_abs) {
        max_abs = abs_val;
        max_row = row;
      }
    }
    if (eigenvectors[max_row][col] < 0) {
      for (let row = 0; row < n; row++) {
        eigenvectors[row][col] = -eigenvectors[row][col];
      }
    }
  }

  // Sort by ascending eigenvalue (should already be sorted, but ensure)
  const sorted = eigenvalues.map((v, i) => ({ v, i }));
  sorted.sort((a, b) => a.v - b.v);

  const sorted_values = sorted.map((p) => p.v);
  const sorted_vectors: number[][] = Array.from({ length: n }, () => new Array(use_k));
  for (let r = 0; r < use_k; r++) {
    const src_col = sorted[r].i;
    for (let row = 0; row < n; row++) {
      sorted_vectors[row][r] = eigenvectors[row][src_col];
    }
  }

  return { eigenvalues: sorted_values, eigenvectors: sorted_vectors };
}


/**
 * Implicit QL algorithm for symmetric tridiagonal matrices.
 *
 * Follows the Numerical Recipes tqli algorithm (Golub & Van Loan §8.3.3).
 * Each outer iteration drives one eigenvalue to convergence via implicit
 * QL steps with Wilkinson shift.
 *
 * @param diagonal     Main diagonal entries (length m)
 * @param off_diagonal  Sub-diagonal entries (length m-1)
 * @returns            Eigenvalues and column-wise eigenvectors
 */
function tridiagonal_ql(
  diagonal: number[],
  off_diagonal: number[],
): { values: number[]; vectors: number[][] } {
  const m = diagonal.length;

  if (m === 0) {
    return { values: [], vectors: [] };
  }

  if (m === 1) {
    return { values: [diagonal[0]], vectors: [[1]] };
  }

  const d = new Float64Array(diagonal);
  const e = new Float64Array(m);
  for (let i = 0; i < off_diagonal.length; i++) e[i] = off_diagonal[i];
  e[m - 1] = 0;

  // Eigenvector accumulator (identity)
  const V: number[][] = Array.from({ length: m }, (_, i) =>
    Array.from({ length: m }, (_, j) => (i === j ? 1 : 0)),
  );

  for (let l = 0; l < m; l++) {
    let iter = 0;

    while (true) {
      // Find smallest mm >= l such that e[mm] is negligible
      let mm: number;
      for (mm = l; mm < m - 1; mm++) {
        const dd = Math.abs(d[mm]) + Math.abs(d[mm + 1]);
        if (Math.abs(e[mm]) <= Number.EPSILON * dd) break;
      }

      if (mm === l) break; // Eigenvalue l has converged

      if (iter++ >= 30) {
        console.warn(
          `[lanczos] Tridiagonal QL did not converge for eigenvalue ${l} after 30 iterations.`,
        );
        break;
      }

      // Wilkinson shift from the bottom 2×2 submatrix of the unreduced block
      let g = (d[l + 1] - d[l]) / (2 * e[l]);
      let r = Math.sqrt(g * g + 1);
      g = d[mm] - d[l] + e[l] / (g + (g >= 0 ? r : -r));

      let s = 1;
      let c = 1;
      let p = 0;

      // Chase the bulge from row mm-1 down to row l
      for (let i = mm - 1; i >= l; i--) {
        const f = s * e[i];
        const b = c * e[i];

        // Givens rotation to eliminate the bulge
        r = Math.sqrt(f * f + g * g);
        e[i + 1] = r;

        if (r === 0) {
          // Underflow — deflate and restart this QL step
          d[i + 1] -= p;
          e[mm] = 0;
          break;
        }

        s = f / r;
        c = g / r;

        g = d[i + 1] - p;
        r = (d[i] - g) * s + 2 * c * b;
        p = s * r;
        d[i + 1] = g + p;
        g = c * r - b; // Critical: update g for next rotation / final e[l]

        // Accumulate eigenvector rotations
        for (let k = 0; k < m; k++) {
          const temp = V[k][i + 1];
          V[k][i + 1] = s * V[k][i] + c * temp;
          V[k][i] = c * V[k][i] - s * temp;
        }
      }

      d[l] -= p;
      e[l] = g;  // Residual off-diagonal from final rotation
      e[mm] = 0;
    }
  }

  const values = Array.from(d);
  return { values, vectors: V };
}

