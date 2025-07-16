import * as tf from "@tensorflow/tfjs-node";

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
export function normalised_laplacian(A: tf.Tensor2D): tf.Tensor2D {
  return tf.tidy(() => {
    const n = A.shape[0];
    const deg = degree_vector(A); // (n)

    // d^{-1/2} – set entries with deg == 0 to 0 (avoids division by zero)
    const invSqrt = tf.where(
      deg.equal(0),
      tf.zerosLike(deg),
      deg.pow(-0.5),
    ) as tf.Tensor1D; // (n)

    // Build outer product invSqrt[:,None] * invSqrt[None,:]  (n,n)
    const diagCol = invSqrt.reshape([n, 1]);
    const diagRow = invSqrt.reshape([1, n]);
    const scaling = diagCol.matMul(diagRow) as tf.Tensor2D; // (n,n)

    const scaledAffinity = A.mul(scaling) as tf.Tensor2D;

    // L = I - scaledAffinity
    const I = tf.eye(n);
    return I.sub(scaledAffinity) as tf.Tensor2D;
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
    maxIterations = 100,
    tolerance = 1e-10,
  }: { maxIterations?: number; tolerance?: number } = {},
): { eigenvalues: number[]; eigenvectors: number[][] } {
  if (matrix.shape.length !== 2 || matrix.shape[0] !== matrix.shape[1]) {
    throw new Error("Input tensor must be square (n × n).");
  }

  // Convert to regular JS arrays for numeric processing.
  const A = matrix.arraySync() as number[][];
  const n = A.length;

  // Deep copy to avoid mutating the original matrix.
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

  const eigenvalues: number[] = D.map((row, i) => row[i]);

  // Re-order eigenvalues (and corresponding eigenvectors) ascending to match
  // common linear algebra library conventions.
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
  const { eigenvalues, eigenvectors } = jacobi_eigen_decomposition(matrix);

  // Pair values with indices → sort ascending
  const indexed = eigenvalues.map((val, idx) => ({ val, idx }));
  indexed.sort((a, b) => a.val - b.val);

  const n = eigenvalues.length;
  const selected: number[][] = Array.from({ length: n }, () => new Array(k));

  for (let col = 0; col < k; col++) {
    for (let row = 0; row < n; row++) {
      selected[row][col] = eigenvectors[row][col];
    }
  }

  return tf.tensor2d(selected, [n, k], "float32");
}
