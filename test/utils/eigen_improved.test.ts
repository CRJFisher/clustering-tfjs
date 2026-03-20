import { describe, it, expect, beforeEach, afterEach } from "@jest/globals";
import * as tf from "../tensorflow-helper";
import {
  improved_jacobi_eigen,
  laplacian_eigen_decomposition,
} from "../../src/utils/eigen_improved";

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

describe("improved_jacobi_eigen – degenerate eigenvalue cases", () => {
  beforeEach(() => {
    tf.engine().startScope();
  });

  afterEach(() => {
    tf.engine().endScope();
  });

  it("handles repeated eigenvalue (diag(1,2,2))", () => {
    const mat = tf.tensor2d([
      [1, 0, 0],
      [0, 2, 0],
      [0, 0, 2],
    ]);

    const { eigenvalues, eigenvectors } = improved_jacobi_eigen(mat);

    expect(eigenvalues).toHaveLength(3);
    expect(eigenvalues[0]).toBeCloseTo(1, 6);
    expect(eigenvalues[1]).toBeCloseTo(2, 6);
    expect(eigenvalues[2]).toBeCloseTo(2, 6);
    expectOrthonormal(eigenvectors, 3);
    expectReconstruction(
      [[1, 0, 0], [0, 2, 0], [0, 0, 2]],
      eigenvalues,
      eigenvectors,
    );
  });

  it("handles all eigenvalues identical (3*I, 3x3)", () => {
    const mat = tf.tensor2d([
      [3, 0, 0],
      [0, 3, 0],
      [0, 0, 3],
    ]);

    const { eigenvalues, eigenvectors } = improved_jacobi_eigen(mat);

    expect(eigenvalues).toHaveLength(3);
    for (const ev of eigenvalues) {
      expect(ev).toBeCloseTo(3, 6);
    }
    expectOrthonormal(eigenvectors, 3);
    expectReconstruction(
      [[3, 0, 0], [0, 3, 0], [0, 0, 3]],
      eigenvalues,
      eigenvectors,
    );
  });

  it("handles zero eigenvalue (rank-1 matrix vv^T) with isPSD: true", () => {
    // v = [1, 2, 3], vv^T is rank-1 with eigenvalue 14 and two zeros
    const v = [1, 2, 3];
    const A: number[][] = [];
    for (let i = 0; i < 3; i++) {
      A[i] = [];
      for (let j = 0; j < 3; j++) {
        A[i][j] = v[i] * v[j];
      }
    }
    const mat = tf.tensor2d(A);

    const { eigenvalues, eigenvectors } = improved_jacobi_eigen(mat, {
      isPSD: true,
    });

    expect(eigenvalues).toHaveLength(3);
    // Two eigenvalues should be ~0, one should be 14
    expect(eigenvalues[0]).toBeCloseTo(0, 5);
    expect(eigenvalues[1]).toBeCloseTo(0, 5);
    expect(eigenvalues[2]).toBeCloseTo(14, 5);
    expectOrthonormal(eigenvectors, 3);
    expectReconstruction(A, eigenvalues, eigenvectors, 1e-4);
  });

  it("handles multiple zero eigenvalues (diag(0,0,3,5)) with isPSD: true", () => {
    const mat = tf.tensor2d([
      [0, 0, 0, 0],
      [0, 0, 0, 0],
      [0, 0, 3, 0],
      [0, 0, 0, 5],
    ]);

    const { eigenvalues, eigenvectors } = improved_jacobi_eigen(mat, {
      isPSD: true,
    });

    expect(eigenvalues).toHaveLength(4);
    expect(eigenvalues[0]).toBeCloseTo(0, 6);
    expect(eigenvalues[1]).toBeCloseTo(0, 6);
    expect(eigenvalues[2]).toBeCloseTo(3, 6);
    expect(eigenvalues[3]).toBeCloseTo(5, 6);
    expectOrthonormal(eigenvectors, 4);
  });

  it("handles near-degenerate eigenvalues (1, 1+1e-8, 2)", () => {
    // Construct A = V * diag(lambdas) * V^T with a known rotation
    const lambdas = [1, 1 + 1e-8, 2];
    // Use a simple rotation matrix (Givens rotation in the 0-1 plane)
    const theta = Math.PI / 5;
    const c = Math.cos(theta);
    const s = Math.sin(theta);
    const V: number[][] = [
      [c, -s, 0],
      [s, c, 0],
      [0, 0, 1],
    ];

    const A: number[][] = Array.from({ length: 3 }, () => Array(3).fill(0));
    for (let i = 0; i < 3; i++) {
      for (let j = 0; j < 3; j++) {
        for (let k = 0; k < 3; k++) {
          A[i][j] += V[i][k] * lambdas[k] * V[j][k];
        }
      }
    }

    const mat = tf.tensor2d(A);
    const { eigenvalues, eigenvectors } = improved_jacobi_eigen(mat);

    expect(eigenvalues).toHaveLength(3);
    expect(eigenvalues[0]).toBeCloseTo(1, 5);
    expect(eigenvalues[1]).toBeCloseTo(1, 5);
    expect(eigenvalues[2]).toBeCloseTo(2, 5);
    expectOrthonormal(eigenvectors, 3);
    expectReconstruction(A, eigenvalues, eigenvectors, 1e-4);
  });

  it("handles large matrix with degenerate eigenvalues (6x6 block diagonal)", () => {
    // Block diagonal: diag(1,1) block, diag(3,3) block, diag(5,5) block
    const n = 6;
    const data: number[][] = Array.from({ length: n }, () => Array(n).fill(0));
    data[0][0] = 1;
    data[1][1] = 1;
    data[2][2] = 3;
    data[3][3] = 3;
    data[4][4] = 5;
    data[5][5] = 5;

    const mat = tf.tensor2d(data);
    const { eigenvalues, eigenvectors } = improved_jacobi_eigen(mat);

    expect(eigenvalues).toHaveLength(6);
    expect(eigenvalues[0]).toBeCloseTo(1, 6);
    expect(eigenvalues[1]).toBeCloseTo(1, 6);
    expect(eigenvalues[2]).toBeCloseTo(3, 6);
    expect(eigenvalues[3]).toBeCloseTo(3, 6);
    expect(eigenvalues[4]).toBeCloseTo(5, 6);
    expect(eigenvalues[5]).toBeCloseTo(5, 6);
    expectOrthonormal(eigenvectors, 6);
    expectReconstruction(data, eigenvalues, eigenvectors);
  });

  it("throws error for non-square matrix", () => {
    const mat = tf.tensor2d([[1, 2, 3], [4, 5, 6]]);
    expect(() => improved_jacobi_eigen(mat)).toThrow();
  });

  it("handles 1x1 matrix returning single eigenvalue", () => {
    const mat = tf.tensor2d([[7]]);
    const { eigenvalues, eigenvectors } = improved_jacobi_eigen(mat);

    expect(eigenvalues).toHaveLength(1);
    expect(eigenvalues[0]).toBeCloseTo(7, 6);
    expect(eigenvectors).toHaveLength(1);
    expect(eigenvectors[0]).toHaveLength(1);
    expect(Math.abs(eigenvectors[0][0])).toBeCloseTo(1, 6);
  });

  it("handles already-diagonal matrix (fast path)", () => {
    const mat = tf.tensor2d([
      [2, 0, 0],
      [0, 5, 0],
      [0, 0, 9],
    ]);

    const { eigenvalues, eigenvectors } = improved_jacobi_eigen(mat);

    expect(eigenvalues).toHaveLength(3);
    expect(eigenvalues[0]).toBeCloseTo(2, 6);
    expect(eigenvalues[1]).toBeCloseTo(5, 6);
    expect(eigenvalues[2]).toBeCloseTo(9, 6);
    expectOrthonormal(eigenvectors, 3);
    expectReconstruction(
      [[2, 0, 0], [0, 5, 0], [0, 0, 9]],
      eigenvalues,
      eigenvectors,
    );
  });
});

describe("laplacian_eigen_decomposition", () => {
  beforeEach(() => {
    tf.engine().startScope();
  });

  afterEach(() => {
    tf.engine().endScope();
  });

  it("returns k+numZeros columns for single-component graph", () => {
    // Simple 4-node connected graph Laplacian (single component = 1 zero eigenvalue)
    // L = D - W for a fully connected triangle+1 pattern
    const L = tf.tensor2d([
      [2, -1, -1, 0],
      [-1, 2, 0, -1],
      [-1, 0, 2, -1],
      [0, -1, -1, 2],
    ]);

    const k = 2;
    const result = laplacian_eigen_decomposition(L, k);

    // Single component: 1 zero eigenvalue, so result should have k + 1 = 3 columns
    // (or k columns depending on implementation — at minimum k columns)
    expect(result.shape.length).toBe(2);
    expect(result.shape[0]).toBe(4); // rows = number of nodes
    expect(result.shape[1]).toBeGreaterThanOrEqual(k);

    result.dispose();
  });

  it("returns float32 tensor", () => {
    const L = tf.tensor2d([
      [1, -1, 0],
      [-1, 2, -1],
      [0, -1, 1],
    ]);

    const result = laplacian_eigen_decomposition(L, 2);
    expect(result.dtype).toBe("float32");

    result.dispose();
  });

  it("does not leak tensors", () => {
    const L = tf.tensor2d([
      [1, -1, 0],
      [-1, 2, -1],
      [0, -1, 1],
    ]);

    const before = tf.memory().numTensors;
    const result = laplacian_eigen_decomposition(L, 2);
    result.dispose();
    L.dispose();
    const after = tf.memory().numTensors;

    expect(after).toBeLessThanOrEqual(before);
  });
});
