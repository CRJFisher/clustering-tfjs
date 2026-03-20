import { describe, it, expect, beforeEach, afterEach } from "@jest/globals";
import * as tf from "../tensorflow-helper";
import {
  qr_eigen_decomposition,
  tridiagonal_qr_eigen,
} from "../../src/utils/eigen_qr";

function expectOrthonormal(
  vectors: number[][],
  n: number,
  tol = 1e-6,
): void {
  for (let i = 0; i < n; i++) {
    let selfDot = 0;
    for (let k = 0; k < n; k++) selfDot += vectors[k][i] * vectors[k][i];
    expect(Math.abs(selfDot - 1)).toBeLessThan(tol);
    for (let j = i + 1; j < n; j++) {
      let dot = 0;
      for (let k = 0; k < n; k++) dot += vectors[k][i] * vectors[k][j];
      expect(Math.abs(dot)).toBeLessThan(tol);
    }
  }
}

function expectReconstruction(
  A: number[][],
  eigenvalues: number[],
  eigenvectors: number[][],
  tol = 1e-6,
): void {
  const n = A.length;
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      let reconstructed = 0;
      for (let k = 0; k < n; k++) {
        reconstructed += eigenvectors[i][k] * eigenvalues[k] * eigenvectors[j][k];
      }
      expect(Math.abs(reconstructed - A[i][j])).toBeLessThan(tol);
    }
  }
}

describe("qr_eigen_decomposition – degenerate eigenvalue cases", () => {
  beforeEach(() => {
    tf.engine().startScope();
  });

  afterEach(() => {
    tf.engine().endScope();
  });

  it("handles all eigenvalues identical (5*I, 4x4)", () => {
    const n = 4;
    const data: number[][] = Array.from({ length: n }, (_, i) =>
      Array.from({ length: n }, (_, j) => (i === j ? 5 : 0)),
    );
    const mat = tf.tensor2d(data);

    const { eigenvalues, eigenvectors } = qr_eigen_decomposition(mat);

    expect(eigenvalues).toHaveLength(4);
    for (const ev of eigenvalues) {
      expect(ev).toBeCloseTo(5, 5);
    }
    expectOrthonormal(eigenvectors, 4);
    expectReconstruction(data, eigenvalues, eigenvectors);
  });

  it("handles double degeneracy (diag(1,1,3,3))", () => {
    const data: number[][] = [
      [1, 0, 0, 0],
      [0, 1, 0, 0],
      [0, 0, 3, 0],
      [0, 0, 0, 3],
    ];
    const mat = tf.tensor2d(data);

    const { eigenvalues, eigenvectors } = qr_eigen_decomposition(mat);

    expect(eigenvalues).toHaveLength(4);
    expect(eigenvalues[0]).toBeCloseTo(1, 5);
    expect(eigenvalues[1]).toBeCloseTo(1, 5);
    expect(eigenvalues[2]).toBeCloseTo(3, 5);
    expect(eigenvalues[3]).toBeCloseTo(3, 5);
    expectOrthonormal(eigenvectors, 4);
    expectReconstruction(data, eigenvalues, eigenvectors);
  });

  it("handles triple degeneracy (diag(2,2,2,5))", () => {
    const data: number[][] = [
      [2, 0, 0, 0],
      [0, 2, 0, 0],
      [0, 0, 2, 0],
      [0, 0, 0, 5],
    ];
    const mat = tf.tensor2d(data);

    const { eigenvalues, eigenvectors } = qr_eigen_decomposition(mat);

    expect(eigenvalues).toHaveLength(4);
    expect(eigenvalues[0]).toBeCloseTo(2, 5);
    expect(eigenvalues[1]).toBeCloseTo(2, 5);
    expect(eigenvalues[2]).toBeCloseTo(2, 5);
    expect(eigenvalues[3]).toBeCloseTo(5, 5);
    expectOrthonormal(eigenvectors, 4);
    expectReconstruction(data, eigenvalues, eigenvectors);
  });

  it("handles zero eigenvalue with multiplicity (diag(0,0,1,2))", () => {
    const data: number[][] = [
      [0, 0, 0, 0],
      [0, 0, 0, 0],
      [0, 0, 1, 0],
      [0, 0, 0, 2],
    ];
    const mat = tf.tensor2d(data);

    const { eigenvalues, eigenvectors } = qr_eigen_decomposition(mat);

    expect(eigenvalues).toHaveLength(4);
    expect(eigenvalues[0]).toBeCloseTo(0, 5);
    expect(eigenvalues[1]).toBeCloseTo(0, 5);
    expect(eigenvalues[2]).toBeCloseTo(1, 5);
    expect(eigenvalues[3]).toBeCloseTo(2, 5);
    expectOrthonormal(eigenvectors, 4);
  });

  // Known bug: Wilkinson-shifted QR produces NaN for non-diagonal matrices
  // when the last off-diagonal element equals zero in the shift calculation.
  // These tests document the bug; remove .failing when the solver is fixed.
  it.failing("handles degenerate eigenvalues from non-diagonal matrix", () => {
    // [[3,1,1],[1,3,1],[1,1,3]] has eigenvalues [1, 4, 4]
    // (eigenvalue 1 with eigenvector in null space of A-I, eigenvalue 4 with multiplicity 2)
    const A: number[][] = [
      [3, 1, 1],
      [1, 3, 1],
      [1, 1, 3],
    ];
    const mat = tf.tensor2d(A);

    const { eigenvalues, eigenvectors } = qr_eigen_decomposition(mat);

    expect(eigenvalues).toHaveLength(3);
    // Sorted ascending: should be approximately [1, 4, 4]
    // Allow for either sort order by checking the set
    const sorted = [...eigenvalues].sort((a, b) => a - b);
    expect(sorted[0]).toBeCloseTo(1, 4);
    expect(sorted[1]).toBeCloseTo(4, 4);
    expect(sorted[2]).toBeCloseTo(4, 4);
    expectOrthonormal(eigenvectors, 3);
    expectReconstruction(A, eigenvalues, eigenvectors, 1e-4);
  });

  it.failing("reconstructs a general symmetric matrix correctly", () => {
    // A symmetric matrix with known structure
    const A: number[][] = [
      [4, 2, 1],
      [2, 5, 3],
      [1, 3, 6],
    ];
    const mat = tf.tensor2d(A);

    const { eigenvalues, eigenvectors } = qr_eigen_decomposition(mat);

    expect(eigenvalues).toHaveLength(3);
    expectOrthonormal(eigenvectors, 3);
    expectReconstruction(A, eigenvalues, eigenvectors, 1e-4);
  });

  it.failing("solves known 2x2 case ([[2,1],[1,2]] has eigenvalues [1,3])", () => {
    const A: number[][] = [
      [2, 1],
      [1, 2],
    ];
    const mat = tf.tensor2d(A);

    const { eigenvalues, eigenvectors } = qr_eigen_decomposition(mat);

    expect(eigenvalues).toHaveLength(2);
    const sorted = [...eigenvalues].sort((a, b) => a - b);
    expect(sorted[0]).toBeCloseTo(1, 5);
    expect(sorted[1]).toBeCloseTo(3, 5);
    expectOrthonormal(eigenvectors, 2);
    expectReconstruction(A, eigenvalues, eigenvectors, 1e-5);
  });
});

describe("tridiagonal_qr_eigen", () => {
  it("handles full degeneracy (all diagonal equal, off-diagonal zero)", () => {
    const d = [4, 4, 4, 4];
    const e = [0, 0, 0];

    const { eigenvalues, eigenvectors } = tridiagonal_qr_eigen(d, e, true);

    expect(eigenvalues).toHaveLength(4);
    for (const ev of eigenvalues) {
      expect(ev).toBeCloseTo(4, 6);
    }
    expect(eigenvectors).toBeDefined();
    expectOrthonormal(eigenvectors!, 4);
  });

  it("handles single element (d=[5], e=[])", () => {
    const { eigenvalues, eigenvectors } = tridiagonal_qr_eigen([5], [], true);

    expect(eigenvalues).toHaveLength(1);
    expect(eigenvalues[0]).toBeCloseTo(5, 6);
    expect(eigenvectors).toBeDefined();
    expect(eigenvectors!).toHaveLength(1);
    expect(eigenvectors![0]).toHaveLength(1);
    expect(Math.abs(eigenvectors![0][0])).toBeCloseTo(1, 6);
  });

  it("returns undefined eigenvectors when computeVectors=false", () => {
    const d = [1, 2, 3];
    const e = [0.5, 0.5];

    const result = tridiagonal_qr_eigen(d, e, false);

    expect(result.eigenvalues).toHaveLength(3);
    expect(result.eigenvectors).toBeUndefined();
  });

  it("correctly decomposes a known symmetric tridiagonal Toeplitz matrix", () => {
    // Symmetric tridiagonal Toeplitz: diagonal = [2, 2, 2], off-diagonal = [-1, -1]
    // Eigenvalues: 2 - 2*cos(k*pi/(n+1)) for k=1..n
    // For n=3: eigenvalues are 2 - 2*cos(pi/4), 2 - 2*cos(pi/2), 2 - 2*cos(3*pi/4)
    //        = 2 - sqrt(2), 2, 2 + sqrt(2)
    const d = [2, 2, 2];
    const e = [-1, -1];

    const { eigenvalues, eigenvectors } = tridiagonal_qr_eigen(d, e, true);

    expect(eigenvalues).toHaveLength(3);
    const sorted = [...eigenvalues].sort((a, b) => a - b);
    expect(sorted[0]).toBeCloseTo(2 - Math.SQRT2, 5);
    expect(sorted[1]).toBeCloseTo(2, 5);
    expect(sorted[2]).toBeCloseTo(2 + Math.SQRT2, 5);
    expect(eigenvectors).toBeDefined();
    expectOrthonormal(eigenvectors!, 3);
  });
});
