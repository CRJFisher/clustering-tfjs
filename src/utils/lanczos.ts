import * as tf from '../tf-adapter';
import { reorthogonalizeVector } from './orthogonalize';
import { make_random_stream } from './rng/index';

/* -------------------------------------------------------------------------- */
/*                     Lanczos Iterative Eigensolver                          */
/* -------------------------------------------------------------------------- */

/**
 * Options for the Lanczos eigensolver.
 */
export interface LanczosOptions {
  /** Maximum Lanczos subspace size before restart. Default: min(max(2k+20, 4k), n, 200) */
  maxSubspaceSize?: number;
  /** Maximum number of thick restarts. Default: 10 */
  maxRestarts?: number;
  /** Convergence tolerance for Ritz values. Default: 1e-6 */
  convergenceTol?: number;
  /** Random seed for deterministic starting vector. Default: 42 */
  randomSeed?: number;
  /** Whether the matrix is PSD (clamp negative eigenvalues to 0). Default: true */
  isPSD?: boolean;
}

/**
 * Result from the Lanczos eigensolver.
 */
export interface LanczosResult {
  /** k smallest eigenvalues, sorted ascending */
  eigenvalues: number[];
  /** Corresponding eigenvectors as column vectors [n][k] */
  eigenvectors: number[][];
}

/* ---- Float32 numerical constants ---- */
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
 * @param k       Number of smallest eigenpairs to compute (1 ≤ k ≤ n)
 * @param options Configuration options
 * @returns       The k smallest eigenvalues and corresponding eigenvectors
 */
export function lanczos_smallest_eigenpairs(
  matrix: tf.Tensor2D,
  k: number,
  options?: LanczosOptions,
): LanczosResult {
  const n = matrix.shape[0];

  if (!Number.isInteger(k) || k < 1) {
    throw new Error('k must be a positive integer >= 1.');
  }
  if (k > n) {
    throw new Error(`k (${k}) cannot exceed matrix size n (${n}).`);
  }
  if (matrix.shape[0] !== matrix.shape[1]) {
    throw new Error('Input matrix must be square.');
  }

  const {
    maxSubspaceSize: maxSubspaceSizeOpt,
    maxRestarts = 10,
    convergenceTol = CONVERGENCE_TOL_DEFAULT,
    randomSeed = 42,
    isPSD = true,
  } = options ?? {};

  // Subspace size: need enough room beyond k for convergence
  const m_max = maxSubspaceSizeOpt ?? Math.min(Math.max(2 * k + 20, 4 * k), n, 200);

  // Extract the matrix to JS arrays once — avoids repeated GPU→CPU transfers.
  // The matvec A*v will be done in pure JS. For n=5000 this is ~25M entries
  // (200MB), same as the current Jacobi solver.
  const A = matrix.arraySync() as number[][];

  // Standard Lanczos: apply A*v directly.
  // Lanczos converges to extreme eigenvalues (both largest and smallest).
  // We extract the k smallest Ritz values from the tridiagonal eigenproblem.

  const rng = make_random_stream(randomSeed);

  // Generate random starting vector of unit norm
  const q = new Float64Array(n);
  let norm = 0;
  for (let i = 0; i < n; i++) {
    q[i] = rng.rand() - 0.5;
    norm += q[i] * q[i];
  }
  norm = Math.sqrt(norm);
  for (let i = 0; i < n; i++) q[i] /= norm;

  // Lanczos basis vectors (columns of Q), stored as array of Float64Array
  let basisVectors: Float64Array[] = [q];

  // Tridiagonal matrix entries
  let alpha: number[] = []; // diagonal
  let beta: number[] = [];  // sub/super-diagonal

  let convergedEigenvalues: number[] | null = null;
  let convergedEigenvectors: number[][] | null = null;

  for (let restart = 0; restart <= maxRestarts; restart++) {
    const startJ = alpha.length; // resumption point after restart

    for (let j = startJ; j < m_max; j++) {
      const v_j = basisVectors[j];

      // w = A * v_j
      const w = matvec(A, v_j, n);

      // w = w - beta_{j-1} * v_{j-1}
      if (j > 0) {
        const beta_prev = beta[j - 1];
        const v_prev = basisVectors[j - 1];
        for (let i = 0; i < n; i++) w[i] -= beta_prev * v_prev[i];
      }

      // alpha_j = v_j^T * w
      let alpha_j = 0;
      for (let i = 0; i < n; i++) alpha_j += v_j[i] * w[i];
      alpha.push(alpha_j);

      // w = w - alpha_j * v_j
      for (let i = 0; i < n; i++) w[i] -= alpha_j * v_j[i];

      // Full reorthogonalization against all previous basis vectors
      const normBefore = vecNorm(w, n);
      reorthogonalize(w, basisVectors, n);
      const normAfter = vecNorm(w, n);

      // Double Gram-Schmidt if significant cancellation detected
      if (normAfter < NORM_REDUCTION_TRIGGER * normBefore) {
        reorthogonalize(w, basisVectors, n);
      }

      const beta_j = vecNorm(w, n);

      // Check for lucky breakdown (invariant subspace found)
      if (beta_j < BREAKDOWN_TOL) {
        // We've found an invariant subspace. Solve for what we have.
        const { values, vectors } = tridiagonal_ql(
          alpha.slice(),
          beta.slice(0, alpha.length - 1),
        );

        const result = extractSmallest(
          values, vectors, basisVectors, k, n, isPSD,
        );

        if (result.eigenvalues.length >= k) {
          return result;
        }

        // Not enough eigenpairs — generate random restart vector
        // orthogonal to current basis
        const newQ = randomOrthogonalVector(basisVectors, n, rng);
        if (newQ === null) {
          // Entire space spanned — return what we have
          return result;
        }
        beta.push(0);
        basisVectors.push(newQ);
        continue;
      }

      beta.push(beta_j);

      // Normalize and append new basis vector
      const q_new = new Float64Array(n);
      for (let i = 0; i < n; i++) q_new[i] = w[i] / beta_j;
      basisVectors.push(q_new);

      // Check convergence periodically (every 5 iterations after minimum)
      if (j >= 2 * k && (j - startJ) % 5 === 4) {
        const { values, vectors } = tridiagonal_ql(
          alpha.slice(),
          beta.slice(0, alpha.length - 1),
        );

        const converged = checkConvergence(
          values, vectors, beta[beta.length - 1],
          k, convergenceTol,
        );

        if (converged) {
          return extractSmallest(
            values, vectors, basisVectors, k, n, isPSD,
          );
        }
      }
    }

    // Reached max subspace size — perform simple restart
    // Solve tridiagonal eigenproblem
    const { values, vectors } = tridiagonal_ql(
      alpha.slice(),
      beta.slice(0, alpha.length - 1),
    );

    // Check if converged at the boundary
    const converged = checkConvergence(
      values, vectors, beta[beta.length - 1],
      k, convergenceTol,
    );

    if (converged) {
      return extractSmallest(
        values, vectors, basisVectors, k, n, isPSD,
      );
    }

    // Simple restart: use the best Ritz vector as new starting vector
    const restartResult = extractSmallest(
      values, vectors, basisVectors, k, n, isPSD,
    );
    convergedEigenvalues = restartResult.eigenvalues;
    convergedEigenvectors = restartResult.eigenvectors;

    // Reconstruct the best Ritz vector (smallest eigenvalue)
    const m = alpha.length;
    const indices = values.map((v, i) => ({ v, i }));
    indices.sort((a, b) => a.v - b.v);

    const bestIdx = indices[0].i;
    const restartVec = new Float64Array(n);
    for (let row = 0; row < n; row++) {
      let sum = 0;
      for (let col = 0; col < m; col++) {
        sum += basisVectors[col][row] * vectors[col][bestIdx];
      }
      restartVec[row] = sum;
    }
    // Normalize
    const restartNorm = vecNorm(restartVec, n);
    if (restartNorm > BREAKDOWN_TOL) {
      for (let i = 0; i < n; i++) restartVec[i] /= restartNorm;
    }

    // Full reset with best Ritz vector as starting point
    basisVectors = [restartVec];
    alpha = [];
    beta = [];
  }

  // Max restarts exhausted — return best result so far
  if (convergedEigenvalues !== null && convergedEigenvectors !== null) {
    console.warn(
      `[lanczos] Did not fully converge after ${maxRestarts} restarts. Returning best approximation.`,
    );
    return {
      eigenvalues: convergedEigenvalues,
      eigenvectors: convergedEigenvectors,
    };
  }

  // Final attempt: solve whatever tridiagonal we have
  const { values, vectors } = tridiagonal_ql(alpha.slice(), beta.slice(0, alpha.length - 1));
  return extractSmallest(values, vectors, basisVectors, k, n, isPSD);
}

/* -------------------------------------------------------------------------- */
/*                             Internal Helpers                               */
/* -------------------------------------------------------------------------- */

/**
 * Computes A*v (standard matrix-vector product).
 */
function matvec(
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

/** Euclidean norm of a vector. */
function vecNorm(v: Float64Array, n: number): number {
  let sum = 0;
  for (let i = 0; i < n; i++) sum += v[i] * v[i];
  return Math.sqrt(sum);
}

/**
 * Full reorthogonalization: projects out components along all basis vectors.
 * Modifies w in place. Delegates to the shared utility.
 */
const reorthogonalize = reorthogonalizeVector<Float64Array>;

/**
 * Generates a random unit vector orthogonal to all basis vectors.
 * Returns null if the basis already spans the full space.
 */
function randomOrthogonalVector(
  basis: Float64Array[],
  n: number,
  rng: ReturnType<typeof make_random_stream>,
): Float64Array | null {
  // Try several random vectors
  for (let attempt = 0; attempt < 5; attempt++) {
    const v = new Float64Array(n);
    for (let i = 0; i < n; i++) v[i] = rng.rand() - 0.5;

    // Project out all basis components (twice for stability)
    reorthogonalize(v, basis, n);
    reorthogonalize(v, basis, n);

    const norm = vecNorm(v, n);
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
function checkConvergence(
  values: number[],
  vectors: number[][],
  lastBeta: number,
  k: number,
  tol: number,
): boolean {
  const m = values.length;
  if (m < k) return false;

  // Sort by ascending value to get the k smallest
  const indices = values.map((v, i) => ({ v, i }));
  indices.sort((a, b) => a.v - b.v);

  for (let r = 0; r < k; r++) {
    const idx = indices[r].i;
    const lastComponent = Math.abs(vectors[m - 1][idx]);
    const residual = Math.abs(lastBeta * lastComponent);
    const threshold = tol * Math.max(1, Math.abs(values[idx]));

    if (residual > threshold) {
      return false;
    }
  }

  return true;
}

/**
 * Extract the k smallest eigenpairs from the Lanczos decomposition.
 * Reconstructs Ritz vectors from the Lanczos basis and tridiagonal eigenvectors.
 */
function extractSmallest(
  ritzValues: number[],
  ritzVectors: number[][],
  basisVectors: Float64Array[],
  k: number,
  n: number,
  isPSD: boolean,
): LanczosResult {
  const m = ritzValues.length;

  // Sort by ascending eigenvalue to get the k smallest
  const indices = ritzValues.map((v, i) => ({ v, i }));
  indices.sort((a, b) => a.v - b.v);

  const useK = Math.min(k, m);
  const eigenvalues: number[] = [];
  const eigenvectors: number[][] = Array.from({ length: n }, () => new Array(useK));

  for (let r = 0; r < useK; r++) {
    const idx = indices[r].i;

    let lambda = ritzValues[idx];

    // PSD clamping
    if (isPSD && lambda < 0) {
      if (lambda < -PSD_CLAMP_TOL) {
        console.warn(
          `[lanczos] Large negative eigenvalue ${lambda} clamped to 0 (exceeds tolerance ${PSD_CLAMP_TOL}).`,
        );
      }
      lambda = 0;
    }

    eigenvalues.push(lambda);

    // Reconstruct Ritz vector: u = Q * s_idx
    // Q is [n, m_basis] where m_basis = basisVectors.length, s is [m, 1]
    const basisCount = Math.min(basisVectors.length, m);
    for (let row = 0; row < n; row++) {
      let sum = 0;
      for (let col = 0; col < basisCount; col++) {
        sum += basisVectors[col][row] * ritzVectors[col][idx];
      }
      eigenvectors[row][r] = sum;
    }
  }

  // Apply deterministic sign convention: largest-magnitude component positive
  for (let col = 0; col < useK; col++) {
    let maxAbs = 0;
    let maxRow = 0;
    for (let row = 0; row < n; row++) {
      const absVal = Math.abs(eigenvectors[row][col]);
      if (absVal > maxAbs) {
        maxAbs = absVal;
        maxRow = row;
      }
    }
    if (eigenvectors[maxRow][col] < 0) {
      for (let row = 0; row < n; row++) {
        eigenvectors[row][col] = -eigenvectors[row][col];
      }
    }
  }

  // Sort by ascending eigenvalue (should already be sorted, but ensure)
  const sorted = eigenvalues.map((v, i) => ({ v, i }));
  sorted.sort((a, b) => a.v - b.v);

  const sortedValues = sorted.map((p) => p.v);
  const sortedVectors: number[][] = Array.from({ length: n }, () => new Array(useK));
  for (let r = 0; r < useK; r++) {
    const srcCol = sorted[r].i;
    for (let row = 0; row < n; row++) {
      sortedVectors[row][r] = eigenvectors[row][srcCol];
    }
  }

  return { eigenvalues: sortedValues, eigenvectors: sortedVectors };
}

/* -------------------------------------------------------------------------- */
/*                  Corrected Tridiagonal QL Eigensolver                      */
/* -------------------------------------------------------------------------- */

/**
 * Implicit QL algorithm for symmetric tridiagonal matrices.
 *
 * Follows the Numerical Recipes tqli algorithm (Golub & Van Loan §8.3.3).
 * Each outer iteration drives one eigenvalue to convergence via implicit
 * QL steps with Wilkinson shift.
 *
 * @param diagonal     Main diagonal entries (length m)
 * @param offDiagonal  Sub-diagonal entries (length m-1)
 * @returns            Eigenvalues and column-wise eigenvectors
 */
function tridiagonal_ql(
  diagonal: number[],
  offDiagonal: number[],
): { values: number[]; vectors: number[][] } {
  const m = diagonal.length;

  if (m === 0) {
    return { values: [], vectors: [] };
  }

  if (m === 1) {
    return { values: [diagonal[0]], vectors: [[1]] };
  }

  // Clone arrays
  const d = new Float64Array(diagonal);
  const e = new Float64Array(m);
  for (let i = 0; i < offDiagonal.length; i++) e[i] = offDiagonal[i];
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

  // Convert to regular arrays
  const values = Array.from(d);
  return { values, vectors: V };
}

