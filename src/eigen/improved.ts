import * as tf from '../backend/adapter';

/**
 * Cyclic sweeps guarantee convergence; adaptive threshold reduction avoids wasted
 * iterations once off-diagonal elements are already small. With `is_psd: true`,
 * tiny negative eigenvalues produced by floating-point error are clamped to zero —
 * appropriate for graph Laplacians where exact zeros represent connected components.
 */
export function improved_jacobi_eigen(
  matrix: tf.Tensor2D,
  {
    max_iterations = 3000,
    tolerance = 1e-14,
    is_psd = false,
  }: { max_iterations?: number; tolerance?: number; is_psd?: boolean } = {},
): { eigenvalues: number[]; eigenvectors: number[][] } {
  if (matrix.shape.length !== 2 || matrix.shape[0] !== matrix.shape[1]) {
    throw new Error('Input tensor must be square (n × n).');
  }

  const A = matrix.arraySync() as number[][];
  const n = A.length;

  const is_approximately_diagonal = (): boolean => {
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        if (i === j) continue;
        if (Math.abs(A[i][j]) > tolerance * 10) return false;
      }
    }
    return true;
  };

  if (is_approximately_diagonal()) {
    const diag = A.map((row, i) => row[i]);
    const V = A.map((_, i) =>
      Array.from({ length: n }, (_, j) => (i === j ? 1 : 0)),
    );
    return { eigenvalues: diag, eigenvectors: V };
  }

  const D: number[][] = A.map((row) => [...row]);

  const V: number[][] = Array.from({ length: n }, (_, i) =>
    Array.from({ length: n }, (_, j) => (i === j ? 1 : 0)),
  );

  const off_diag_norm = (M: number[][]): number => {
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

  const matrix_norm = Math.sqrt(
    A.reduce((sum, row) => sum + row.reduce((s, v) => s + v * v, 0), 0),
  );
  let threshold = tolerance * matrix_norm;

  while (sweep < max_iterations && changed) {
    changed = false;

    for (let p = 0; p < n - 1; p++) {
      for (let q = p + 1; q < n; q++) {
        const a_pq = D[p][q];

        if (Math.abs(a_pq) < threshold) continue;

        changed = true;
        const a_pp = D[p][p];
        const a_qq = D[q][q];

        const diff = a_qq - a_pp;
        let t: number;

        const abs_diff = Math.abs(diff);
        if (
          abs_diff <=
          Number.EPSILON * Math.max(Math.abs(a_pp), Math.abs(a_qq), 1)
        ) {
          // Equal or nearly-equal diagonal elements: optimal rotation is pi/4
          // t = sign(a_pq) following standard Jacobi (Golub & Van Loan)
          t = a_pq >= 0 ? 1 : -1;
        } else if (Math.abs(a_pq) < abs_diff * 1e-15) {
          // Very small angle - use approximation (diff is safely nonzero)
          t = a_pq / diff;
        } else {
          const theta = diff / (2 * a_pq);
          t = 1 / (Math.abs(theta) + Math.sqrt(1 + theta * theta));
          if (theta < 0) t = -t;
        }

        const c = 1 / Math.sqrt(1 + t * t);
        const s = t * c;
        const tau = s / (1 + c);

        D[p][p] = a_pp - t * a_pq;
        D[q][q] = a_qq + t * a_pq;
        D[p][q] = D[q][p] = 0;

        for (let j = 0; j < p; j++) {
          const g = D[j][p];
          const h = D[j][q];
          D[j][p] = D[p][j] = g - s * (h + g * tau);
          D[j][q] = D[q][j] = h + s * (g - h * tau);
        }

        for (let j = p + 1; j < q; j++) {
          const g = D[p][j];
          const h = D[j][q];
          D[p][j] = D[j][p] = g - s * (h + g * tau);
          D[j][q] = D[q][j] = h + s * (g - h * tau);
        }

        for (let j = q + 1; j < n; j++) {
          const g = D[p][j];
          const h = D[q][j];
          D[p][j] = D[j][p] = g - s * (h + g * tau);
          D[q][j] = D[j][q] = h + s * (g - h * tau);
        }

        for (let j = 0; j < n; j++) {
          const g = V[j][p];
          const h = V[j][q];
          V[j][p] = g - s * (h + g * tau);
          V[j][q] = h + s * (g - h * tau);
        }
      }
    }

    sweep++;

    if (sweep % 5 === 0) {
      const current_norm = off_diag_norm(D);
      if (current_norm < threshold * n) {
        threshold *= 0.1;
      }
    }
  }

  if (sweep === max_iterations) {
    console.warn(
      `Improved Jacobi solver reached max iterations (${max_iterations}). ` +
        `Final off-diagonal norm: ${off_diag_norm(D).toExponential(3)}`,
    );
  }

  let eigenvalues: number[] = D.map((row, i) => row[i]);

  // Post-processing for PSD matrices: clamp only negative eigenvalues to zero.
  // Small positive eigenvalues (e.g. 1e-9) are legitimate and represent
  // weakly connected components or near-separability in the graph.
  if (is_psd) {
    const negative_tolerance = tolerance * 100;
    eigenvalues = eigenvalues.map((v) => {
      if (v < 0) {
        if (v < -negative_tolerance) {
          console.warn(
            `[spectral] Large negative eigenvalue ${v} in PSD matrix (exceeds tolerance ${negative_tolerance}). Clamping to 0.`,
          );
        }
        return 0;
      }
      return v;
    });
  }

  const indexed = eigenvalues.map((val, idx) => ({ val, idx }));
  indexed.sort((a, b) => a.val - b.val);

  const sorted_values = indexed.map((p) => p.val);
  const sorted_vectors: number[][] = Array.from(
    { length: n },
    () => new Array(n),
  );

  for (let new_idx = 0; new_idx < n; new_idx++) {
    const old_idx = indexed[new_idx].idx;
    for (let row = 0; row < n; row++) {
      sorted_vectors[row][new_idx] = V[row][old_idx];
    }
  }

  let max_orth_error = 0;
  let max_norm_error = 0;
  for (let i = 0; i < n; i++) {
    let self_dot = 0;
    for (let k = 0; k < n; k++) {
      self_dot += sorted_vectors[k][i] * sorted_vectors[k][i];
    }
    max_norm_error = Math.max(max_norm_error, Math.abs(self_dot - 1));
    for (let j = i + 1; j < n; j++) {
      let dot = 0;
      for (let k = 0; k < n; k++) {
        dot += sorted_vectors[k][i] * sorted_vectors[k][j];
      }
      max_orth_error = Math.max(max_orth_error, Math.abs(dot));
    }
  }
  if (max_orth_error > 1e-6 || max_norm_error > 1e-6) {
    console.warn(
      `[spectral] Eigenvector orthonormality check: max |v_i · v_j| = ${max_orth_error.toExponential(3)}, max ||v_i|| - 1| = ${max_norm_error.toExponential(3)} (threshold 1e-6)`,
    );
  }

  return { eigenvalues: sorted_values, eigenvectors: sorted_vectors };
}

