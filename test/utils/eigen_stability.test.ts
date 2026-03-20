import * as tf from "../tensorflow-helper";

import { improved_jacobi_eigen } from "../../src/utils/eigen_improved";
import {
  qr_eigen_decomposition,
  tridiagonal_qr_eigen,
} from "../../src/utils/eigen_qr";
import { smallest_eigenvectors_with_values } from "../../src/utils/smallest_eigenvectors_with_values";

/* -------------------------------------------------------------------------- */
/*  Helper: check eigenvector orthonormality                                  */
/* -------------------------------------------------------------------------- */
function expectOrthonormal(
  vecs: number[][],
  n: number,
  numCols: number,
  tol = 1e-8,
) {
  for (let i = 0; i < numCols; i++) {
    for (let j = 0; j < numCols; j++) {
      let dot = 0;
      for (let k = 0; k < n; k++) {
        dot += vecs[k][i] * vecs[k][j];
      }
      if (i === j) {
        expect(Math.abs(dot - 1)).toBeLessThan(tol);
      } else {
        expect(Math.abs(dot)).toBeLessThan(tol);
      }
    }
  }
}

/* -------------------------------------------------------------------------- */
/*  Helper: reconstruct A = V diag(lambda) V^T and compare                   */
/* -------------------------------------------------------------------------- */
function expectReconstruction(
  eigenvalues: number[],
  eigenvectors: number[][],
  original: number[][],
  tol = 1e-8,
) {
  const n = eigenvalues.length;
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      let sum = 0;
      for (let k = 0; k < n; k++) {
        sum += eigenvectors[i][k] * eigenvalues[k] * eigenvectors[j][k];
      }
      expect(Math.abs(sum - original[i][j])).toBeLessThan(tol);
    }
  }
}

/* ========================================================================== */
/*  1. Jacobi div-by-zero: a_pp == a_qq                                      */
/* ========================================================================== */
describe("improved_jacobi_eigen – equal diagonal elements", () => {
  it("handles a_pp == a_qq (2×2 case)", () => {
    // [[3, 1], [1, 3]] has eigenvalues 2 and 4
    const M = tf.tensor2d(
      [
        [3, 1],
        [1, 3],
      ],
      [2, 2],
    );
    const { eigenvalues, eigenvectors } = improved_jacobi_eigen(M);

    eigenvalues.forEach((v) => expect(Number.isFinite(v)).toBe(true));
    eigenvectors.flat().forEach((v) => expect(Number.isFinite(v)).toBe(true));

    expect(eigenvalues[0]).toBeCloseTo(2, 10);
    expect(eigenvalues[1]).toBeCloseTo(4, 10);

    expectOrthonormal(eigenvectors, 2, 2);
    M.dispose();
  });

  it("handles all-equal diagonals (4×4 tridiagonal Toeplitz)", () => {
    // All diagonal entries are 2; eigenvalues: 2 - 2cos(kπ/5) for k=1..4
    const M = tf.tensor2d(
      [
        [2, 1, 0, 0],
        [1, 2, 1, 0],
        [0, 1, 2, 1],
        [0, 0, 1, 2],
      ],
      [4, 4],
    );
    const { eigenvalues, eigenvectors } = improved_jacobi_eigen(M);

    eigenvalues.forEach((v) => expect(Number.isFinite(v)).toBe(true));
    eigenvectors.flat().forEach((v) => expect(Number.isFinite(v)).toBe(true));

    const expected = [1, 2, 3, 4].map((k) => 2 - 2 * Math.cos((k * Math.PI) / 5));
    expected.sort((a, b) => a - b);
    for (let i = 0; i < 4; i++) {
      expect(eigenvalues[i]).toBeCloseTo(expected[i], 8);
    }

    expectOrthonormal(eigenvectors, 4, 4);
    M.dispose();
  });

  it("handles normalized Laplacian of complete graph K3", () => {
    // L_norm of K3: eigenvalues 0, 1.5, 1.5
    const L = tf.tensor2d(
      [
        [1, -0.5, -0.5],
        [-0.5, 1, -0.5],
        [-0.5, -0.5, 1],
      ],
      [3, 3],
    );
    const { eigenvalues, eigenvectors } = improved_jacobi_eigen(L, {
      isPSD: true,
    });

    eigenvalues.forEach((v) => expect(Number.isFinite(v)).toBe(true));
    expect(eigenvalues[0]).toBeCloseTo(0, 5);
    expect(eigenvalues[1]).toBeCloseTo(1.5, 5);
    expect(eigenvalues[2]).toBeCloseTo(1.5, 5);

    expectOrthonormal(eigenvectors, 3, 3);
    L.dispose();
  });
});

/* ========================================================================== */
/*  2. PSD clamping: preserves small positives, clamps negatives              */
/* ========================================================================== */
describe("improved_jacobi_eigen – PSD clamping", () => {
  it("preserves small positive eigenvalues", () => {
    // Diagonal matrix with a small positive eigenvalue
    const M = tf.tensor2d(
      [
        [1e-9, 0, 0],
        [0, 1, 0],
        [0, 0, 2],
      ],
      [3, 3],
    );
    const { eigenvalues } = improved_jacobi_eigen(M, { isPSD: true });

    // 1e-9 should NOT be clamped to 0
    expect(eigenvalues[0]).toBeCloseTo(1e-9, 12);
    expect(eigenvalues[0]).toBeGreaterThan(0);

    M.dispose();
  });

  it("clamps negative eigenvalues to zero in PSD mode", () => {
    // Near-PSD matrix that produces a tiny negative eigenvalue
    // [[1, 1+eps], [1+eps, 1]] has eigenvalues 2+eps and -eps
    const eps = 1e-13;
    const M = tf.tensor2d(
      [
        [1, 1 + eps],
        [1 + eps, 1],
      ],
      [2, 2],
    );
    const { eigenvalues } = improved_jacobi_eigen(M, { isPSD: true });

    eigenvalues.forEach((v) => expect(v).toBeGreaterThanOrEqual(0));

    M.dispose();
  });

  it("does not clamp negative eigenvalues without isPSD", () => {
    // Indefinite matrix [[0, 1], [1, 0]] has eigenvalues -1 and +1
    const M = tf.tensor2d(
      [
        [0, 1],
        [1, 0],
      ],
      [2, 2],
    );
    const { eigenvalues } = improved_jacobi_eigen(M, { isPSD: false });

    expect(eigenvalues[0]).toBeCloseTo(-1, 10);
    expect(eigenvalues[1]).toBeCloseTo(1, 10);

    M.dispose();
  });
});

/* ========================================================================== */
/*  3. Tridiagonal QR: e[i] from inner loop computation                       */
/* ========================================================================== */
describe("tridiagonal_qr_eigen – correctness", () => {
  it("computes eigenvalues of simple 3×3 tridiagonal matrix", () => {
    // Tridiagonal: d=[2,2,2], e=[1,1]
    // Same as 3×3 Toeplitz: eigenvalues 2-√2, 2, 2+√2
    const { eigenvalues, eigenvectors } = tridiagonal_qr_eigen(
      [2, 2, 2],
      [1, 1],
    );

    expect(eigenvalues[0]).toBeCloseTo(2 - Math.SQRT2, 8);
    expect(eigenvalues[1]).toBeCloseTo(2, 8);
    expect(eigenvalues[2]).toBeCloseTo(2 + Math.SQRT2, 8);

    expectOrthonormal(eigenvectors!, 3, 3);
  });

  it("handles identity matrix (all zeros off-diagonal)", () => {
    const { eigenvalues } = tridiagonal_qr_eigen([1, 1, 1, 1], [0, 0, 0]);

    eigenvalues.forEach((v) => expect(v).toBeCloseTo(1, 10));
  });

  it("handles 2×2 tridiagonal", () => {
    // d=[3,5], e=[2]: matrix [[3,2],[2,5]]
    // eigenvalues: 4 ± √5
    const { eigenvalues } = tridiagonal_qr_eigen([3, 5], [2]);

    expect(eigenvalues[0]).toBeCloseTo(4 - Math.sqrt(5), 8);
    expect(eigenvalues[1]).toBeCloseTo(4 + Math.sqrt(5), 8);
  });

  it("reconstructs original tridiagonal matrix from eigenpairs", () => {
    const d = [1, 3, 2, 4];
    const e = [0.5, 0.7, 0.3];
    const { eigenvalues, eigenvectors } = tridiagonal_qr_eigen(d, e);

    // Build full tridiagonal matrix
    const n = d.length;
    const T: number[][] = Array.from({ length: n }, () => Array(n).fill(0));
    for (let i = 0; i < n; i++) {
      T[i][i] = d[i];
      if (i < n - 1) {
        T[i][i + 1] = e[i];
        T[i + 1][i] = e[i];
      }
    }

    expectReconstruction(eigenvalues, eigenvectors!, T, 1e-6);
  });
});

/* ========================================================================== */
/*  4. Wilkinson shift: delta==0 b==0 edge case                               */
/* ========================================================================== */
describe("qr_eigen_decomposition – Wilkinson shift NaN", () => {
  it("handles diagonal matrix (delta==0, b==0)", () => {
    const M = tf.tensor2d(
      [
        [2, 0],
        [0, 2],
      ],
      [2, 2],
    );
    const { eigenvalues, eigenvectors } = qr_eigen_decomposition(M);

    eigenvalues.forEach((v) => {
      expect(Number.isFinite(v)).toBe(true);
      expect(v).toBeCloseTo(2, 8);
    });
    eigenvectors.flat().forEach((v) => expect(Number.isFinite(v)).toBe(true));

    M.dispose();
  });

  it("handles identity matrix without NaN", () => {
    const M = tf.eye(3) as tf.Tensor2D;
    const { eigenvalues, eigenvectors } = qr_eigen_decomposition(M);

    eigenvalues.forEach((v) => {
      expect(Number.isFinite(v)).toBe(true);
      expect(v).toBeCloseTo(1, 8);
    });
    eigenvectors.flat().forEach((v) => expect(Number.isFinite(v)).toBe(true));

    M.dispose();
  });

  it("handles matrix with repeated eigenvalues", () => {
    // diag(1, 1, 2)
    const M = tf.tensor2d(
      [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 2],
      ],
      [3, 3],
    );
    const { eigenvalues } = qr_eigen_decomposition(M);

    eigenvalues.forEach((v) => expect(Number.isFinite(v)).toBe(true));
    const sorted = [...eigenvalues].sort((a, b) => a - b);
    expect(sorted[0]).toBeCloseTo(1, 8);
    expect(sorted[1]).toBeCloseTo(1, 8);
    expect(sorted[2]).toBeCloseTo(2, 8);

    M.dispose();
  });
});

/* ========================================================================== */
/*  5. Zero-eigenvalue tolerance                                              */
/* ========================================================================== */
describe("smallest_eigenvectors_with_values – zero-eigenvalue tolerance", () => {
  it("does not treat eigenvalue ~0.005 as zero", () => {
    // Diagonal matrix with eigenvalues [0, 0.005, 1.0, 1.5]
    const M = tf.tensor2d(
      [
        [0, 0, 0, 0],
        [0, 0.005, 0, 0],
        [0, 0, 1.0, 0],
        [0, 0, 0, 1.5],
      ],
      [4, 4],
    );

    const result = smallest_eigenvectors_with_values(M, 2);

    // c = 1 (only eigenvalue 0 is zero with tolerance 1e-7)
    // sliceCols = min(2 + 1, 4) = 3
    expect(result.eigenvectors.shape[1]).toBe(3);

    const eigenData = result.eigenvalues.arraySync() as number[];
    expect(eigenData[0]).toBeCloseTo(0, 7);
    expect(eigenData[1]).toBeCloseTo(0.005, 3);

    result.eigenvectors.dispose();
    result.eigenvalues.dispose();
    M.dispose();
  });

  it("correctly identifies truly zero eigenvalues", () => {
    // Two zero eigenvalues (disconnected components) + two nonzero
    const M = tf.tensor2d(
      [
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 1.0, 0],
        [0, 0, 0, 2.0],
      ],
      [4, 4],
    );

    const result = smallest_eigenvectors_with_values(M, 2);

    // c = 2 (two zero eigenvalues), sliceCols = min(2 + 2, 4) = 4
    expect(result.eigenvectors.shape[1]).toBe(4);

    const eigenData = result.eigenvalues.arraySync() as number[];
    expect(eigenData[0]).toBeCloseTo(0, 7);
    expect(eigenData[1]).toBeCloseTo(0, 7);

    result.eigenvectors.dispose();
    result.eigenvalues.dispose();
    M.dispose();
  });

  it("eigenvalue 1e-6 is not treated as zero", () => {
    const M = tf.tensor2d(
      [
        [0, 0, 0],
        [0, 1e-6, 0],
        [0, 0, 1.5],
      ],
      [3, 3],
    );

    const result = smallest_eigenvectors_with_values(M, 2);

    // c = 1 (only exact 0 is zero; 1e-6 > 1e-7)
    // sliceCols = min(2 + 1, 3) = 3
    expect(result.eigenvectors.shape[1]).toBe(3);

    result.eigenvectors.dispose();
    result.eigenvalues.dispose();
    M.dispose();
  });
});

/* ========================================================================== */
/*  6. Eigenvector orthogonality post-condition                               */
/* ========================================================================== */
describe("eigenvector orthogonality validation", () => {
  it("improved Jacobi produces orthonormal eigenvectors for general symmetric matrix", () => {
    const M = tf.tensor2d(
      [
        [4, 1, 0.5],
        [1, 3, 0.2],
        [0.5, 0.2, 2],
      ],
      [3, 3],
    );
    const { eigenvalues, eigenvectors } = improved_jacobi_eigen(M);

    expectOrthonormal(eigenvectors, 3, 3);
    expectReconstruction(eigenvalues, eigenvectors, M.arraySync() as number[][]);

    M.dispose();
  });

  it("tridiagonal QR produces orthonormal eigenvectors", () => {
    const { eigenvalues, eigenvectors } = tridiagonal_qr_eigen(
      [4, 3, 2, 1],
      [0.5, 0.7, 0.3],
    );

    expectOrthonormal(eigenvectors!, 4, 4, 1e-6);
  });
});
