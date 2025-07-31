import * as tf from "./tensorflow";

import { deterministic_eigenpair_processing } from "./eigen_post";

/* -------------------------------------------------------------------------- */
/*                        Graph Laplacian – core utilities                    */
/* -------------------------------------------------------------------------- */

/**
 * Computes the (row) degree vector for the provided *affinity / similarity*
 * matrix.
 *
 * The input must be a **square** `tf.Tensor2D` whose entries represent edge
 * weights `A[i,j]` of an undirected graph.  For *k*-NN graphs the matrix is
 * expected to be symmetrised (`A = max(A, Aᵀ)`).
 *
 * The returned tensor is a 1-D vector where `deg[i] = Σ_j A[i,j]`.
 */
export function degree_vector(A: tf.Tensor2D): tf.Tensor1D {
  if (A.shape.length !== 2 || A.shape[0] !== A.shape[1]) {
    throw new Error("Affinity matrix must be square (n × n).");
  }

  // Sum along axis 1 (rows) – (n)
  return tf.tidy(() => A.sum(1) as tf.Tensor1D);
}

/**
 * Computes the *symmetric normalised* graph Laplacian
 *
 *     L = I  −  D^{-1/2} · A · D^{-1/2}
 *
 * where
 *   • `A` is the affinity / adjacency matrix
 *   • `D` is the diagonal *degree* matrix with `D[i,i] = Σ_j A[i,j]`
 *
 * Isolated vertices (zero degree) are handled gracefully by treating
 * `D^{-1/2}` as **0** for the corresponding diagonal entry which leaves the
 * row & column of `L` equal to the identity matrix (no connections).
 */
export function normalised_laplacian(A: tf.Tensor2D): tf.Tensor2D;
export function normalised_laplacian(A: tf.Tensor2D, returnDiag: true): { laplacian: tf.Tensor2D; sqrtDegrees: tf.Tensor1D };
export function normalised_laplacian(A: tf.Tensor2D, returnDiag = false): tf.Tensor2D | { laplacian: tf.Tensor2D; sqrtDegrees: tf.Tensor1D } {
  return tf.tidy(() => {
    const n = A.shape[0];
    
    // First, zero out the diagonal of A to match scipy behavior
    // "Diagonal entries of the input adjacency matrix are ignored and
    // replaced with zeros for the purpose of normalization"
    const diagMask = tf.sub(1, tf.eye(n));
    const A_no_diag = A.mul(diagMask) as tf.Tensor2D;
    
    const deg = degree_vector(A_no_diag); // (n)

    // d^{-1/2} – set entries with deg == 0 to 1 (for isolated nodes)
    const invSqrt = tf.where(
      deg.equal(0),
      tf.onesLike(deg),
      deg.pow(-0.5),
    ) as tf.Tensor1D; // (n)

    // Build outer product invSqrt[:,None] * invSqrt[None,:]  (n,n)
    const diagCol = invSqrt.reshape([n, 1]);
    const diagRow = invSqrt.reshape([1, n]);
    const scaling = diagCol.matMul(diagRow) as tf.Tensor2D; // (n,n)

    // Scale the affinity matrix (with diagonal already zeroed)
    const scaledAffinity = A_no_diag.mul(scaling) as tf.Tensor2D;

    // L = I - scaledAffinity
    // This ensures diagonal entries are exactly 1 for non-isolated nodes
    const I = tf.eye(n);
    const laplacian = I.sub(scaledAffinity) as tf.Tensor2D;
    
    if (returnDiag) {
      // Return both Laplacian and sqrt(degrees) for eigenvector recovery
      // Note: we return invSqrt which is D^(-1/2), so for recovery we need to divide by it
      return { 
        laplacian: laplacian,
        sqrtDegrees: invSqrt
      };
    }
    
    return laplacian;
  });
}

/* -------------------------------------------------------------------------- */
/*                        Eigendecomposition (Jacobi)                         */
/* -------------------------------------------------------------------------- */

/**
 * Lightweight *Jacobi* eigen-decomposition for **symmetric** matrices.
 *
 * TensorFlow.js currently does not expose an `eig` kernel for Node/GPU
 * back-ends.  For the small matrix sizes typical in unit tests and many
 * practical spectral-clustering scenarios (n ≤ few hundred) a pure
 * JavaScript Jacobi solver provides sufficiently fast and stable results
 * without adding an external dependency.
 *
 * The implementation is deliberately kept simple and therefore **not**
 * optimised for large dense matrices.  It should *only* be used on matrices
 * of moderate size.
 */
export function jacobi_eigen_decomposition(
  matrix: tf.Tensor2D,
  {
    maxIterations = 2000,
    tolerance = 1e-12,
  }: { maxIterations?: number; tolerance?: number } = {},
): { eigenvalues: number[]; eigenvectors: number[][] } {
  if (matrix.shape.length !== 2 || matrix.shape[0] !== matrix.shape[1]) {
    throw new Error("Input tensor must be square (n × n).");
  }

  // Convert to regular JS arrays for numeric processing.
  const A = matrix.arraySync() as number[][];
  const n = A.length;

  // ---------------------------------------------------------------------
  // Fast-path: if the matrix is already (almost) diagonal we can return
  //            immediately.  This situation commonly arises for
  //            *disconnected* graphs where the *normalised* Laplacian is
  //            the identity matrix or block-diagonal with very small
  //            off-diagonal entries (numerical noise).
  // ---------------------------------------------------------------------

  const isApproximatelyDiagonal = (): boolean => {
    const nDiagTolerance = tolerance * 10;
    for (let i = 0; i < matrix.shape[0]; i++) {
      for (let j = 0; j < matrix.shape[0]; j++) {
        if (i === j) continue;
        if (Math.abs(A[i][j]) > nDiagTolerance) return false;
      }
    }
    return true;
  };

  if (isApproximatelyDiagonal()) {
    warn(
      "Input matrix is (almost) diagonal – skipped iterative Jacobi rotations for efficiency.",
    );
    const diag = A.map((row, i) => row[i]);
    const V = A.map((_, i) =>
      Array.from({ length: matrix.shape[0] }, (_, j) => (i === j ? 1 : 0)),
    );
    return { eigenvalues: diag, eigenvectors: V };
  }

  // Deep copy to avoid mutating the original matrix during rotations.
  const D: number[][] = Array.from({ length: n }, (_, i) =>
    Array.from({ length: n }, (_, j) => A[i][j]),
  );

  // Eigenvector accumulator (initially identity)
  const V: number[][] = Array.from({ length: n }, (_, i) =>
    Array.from({ length: n }, (_, j) => (i === j ? 1 : 0)),
  );

  const offDiag = (M: number[][]): number => {
    let sum = 0;
    for (let i = 0; i < n; i++) {
      for (let j = i + 1; j < n; j++) {
        const val = M[i][j];
        sum += val * val;
      }
    }
    return Math.sqrt(sum);
  };

  let iter = 0;
  while (iter < maxIterations && offDiag(D) > tolerance) {
    // Find largest off-diagonal element (by absolute value)
    let p = 0;
    let q = 1;
    let maxVal = Math.abs(D[p][q]);
    for (let i = 0; i < n; i++) {
      for (let j = i + 1; j < n; j++) {
        const val = Math.abs(D[i][j]);
        if (val > maxVal) {
          maxVal = val;
          p = i;
          q = j;
        }
      }
    }

    if (maxVal < tolerance) break;

    const a_pp = D[p][p];
    const a_qq = D[q][q];
    const a_pq = D[p][q];

    // Compute rotation angle
    // When a_pq is extremely small we risk dividing by ~0 which would blow
    // up `tau`.  In that situation the Jacobi rotation angle can be set to
    // 0 because the off-diagonal entry is already tiny compared with the
    // tolerance threshold (and will therefore be eliminated in the next
    // iteration criterion).  This guards against `Infinity` / `NaN`
    // propagation that would otherwise terminate the algorithm.
    if (Math.abs(a_pq) < tolerance) {
      warn(
        `Jacobi pivot below tolerance (|a_pq|≈${Math.abs(a_pq)}). Skipping rotation.`,
      );
      D[p][q] = D[q][p] = 0;
      iter += 1;
      continue;
    }

    const tau = (a_qq - a_pp) / (2 * a_pq);
    let t: number;
    if (tau === 0) {
      t = 1;
    } else {
      t = Math.sign(tau) / (Math.abs(tau) + Math.sqrt(1 + tau * tau));
    }
    const c = 1 / Math.sqrt(1 + t * t);
    const s = t * c;

    // Rotate rows & columns p and q in D
    for (let k = 0; k < n; k++) {
      const d_pk = D[p][k];
      const d_qk = D[q][k];
      D[p][k] = d_pk * c - d_qk * s;
      D[q][k] = d_pk * s + d_qk * c;
    }

    for (let k = 0; k < n; k++) {
      const d_kp = D[k][p];
      const d_kq = D[k][q];
      D[k][p] = d_kp * c - d_kq * s;
      D[k][q] = d_kp * s + d_kq * c;
    }

    // Manually set symmetric entries we overwrote
    D[p][p] = a_pp * c * c - 2 * a_pq * c * s + a_qq * s * s;
    D[q][q] = a_pp * s * s + 2 * a_pq * c * s + a_qq * c * c;
    D[p][q] = D[q][p] = 0; // by design

    // Update eigenvectors matrix
    for (let k = 0; k < n; k++) {
      const v_kp = V[k][p];
      const v_kq = V[k][q];
      V[k][p] = v_kp * c - v_kq * s;
      V[k][q] = v_kp * s + v_kq * c;
    }

    iter += 1;
  }
  
  if (iter === maxIterations) {
    warn(
      `Jacobi solver did not converge after ${maxIterations} iterations. Final off-diagonal norm: ${offDiag(D)}`,
    );
  }

  let eigenvalues: number[] = D.map((row, i) => row[i]);

  // Clamp very small negative values arising from numerical noise to 0.
  eigenvalues = eigenvalues.map((v) => {
    if (v < 0 && v > -tolerance) {
      warn(
        `Negative eigenvalue ${v} clamped to 0 (within numeric tolerance).`,
      );
      return 0;
    }
    return v;
  });

  // Sort eigen-pairs ascending (deterministic sign handling is applied later
  // in `smallest_eigenvectors` for the subset actually used in the spectral
  // pipeline – avoids unnecessary sign flips for callers that do not care).

  const indexed = eigenvalues.map((val, idx) => ({ val, idx }));
  indexed.sort((a, b) => a.val - b.val);

  const sortedValues = indexed.map((p) => p.val);
  const sortedVectors: number[][] = Array.from({ length: n }, () => new Array(n));

  for (let newIdx = 0; newIdx < n; newIdx++) {
    const oldIdx = indexed[newIdx].idx;
    for (let row = 0; row < n; row++) {
      sortedVectors[row][newIdx] = V[row][oldIdx];
    }
  }

  return { eigenvalues: sortedValues, eigenvectors: sortedVectors };

  /* istanbul ignore next */
  function warn(msg: string): void {
    // Centralised helper – could be swapped for a proper logger later.
    // Users may suppress by overriding console.warn if desired.
    // We add a prefix to make it searchable in logs.
    console.warn(`[spectral] ${msg}`);
  }
}

/**
 * Convenience helper that returns the `k` smallest eigenvectors of the
 * provided symmetric matrix *as a TensorFlow.js tensor* (n × k).
 */
export function smallest_eigenvectors(
  matrix: tf.Tensor2D,
  k: number,
): tf.Tensor2D {
  if (!Number.isInteger(k) || k < 1) {
    throw new Error("k must be a positive integer.");
  }

  return tf.tidy(() => {
    // Import improved solver for better accuracy
    // eslint-disable-next-line @typescript-eslint/no-var-requires
    const { improved_jacobi_eigen } = require("./eigen_improved");
    
    // 1) Full eigendecomposition with improved solver
    // For normalized Laplacians, we know it's PSD
    const { eigenvalues, eigenvectors } = improved_jacobi_eigen(matrix, {
      isPSD: true,
      maxIterations: 3000,
      tolerance: 1e-14,
    });

    // 2) Deterministic ordering & sign fixing (task-12.3.1 helper)
    const { eigenvectors: vecSorted } = deterministic_eigenpair_processing({
      eigenvalues,
      eigenvectors,
    });

    // 3) Determine number of numerically-zero eigenvalues `c` (≤ n).  We must
    //    include *all* corresponding eigenvectors because each represents a
    //    connected component in the affinity graph.  scikit-learn retains
    //    them initially and discards them later after constructing the full
    //    embedding.  We mimic this contract so callers can remove the block
    //    in one go.

    const n = vecSorted.length;
    // For spectral clustering, we want exactly k eigenvectors
    // INCLUDING any with zero eigenvalues (they encode component structure)
    const sliceCols = Math.min(k, n);
    const selected: number[][] = Array.from({ length: n }, () =>
      new Array(sliceCols),
    );

    for (let col = 0; col < sliceCols; col++) {
      for (let row = 0; row < n; row++) {
        selected[row][col] = vecSorted[row][col];
      }
    }

    return tf.tensor2d(selected, [n, sliceCols], "float32");
  });
}
