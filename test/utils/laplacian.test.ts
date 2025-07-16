import * as tf from "@tensorflow/tfjs-node";

import {
  degree_vector,
  normalised_laplacian,
  jacobi_eigen_decomposition,
  smallest_eigenvectors,
} from "../../src";

describe("Graph Laplacian utilities", () => {
  it("computes correct degree vector", () => {
    const A = tf.tensor2d(
      [
        [0, 1, 2],
        [1, 0, 0],
        [2, 0, 0],
      ],
      [3, 3],
    );
    const deg = degree_vector(A).arraySync();
    expect(deg).toEqual([3, 1, 2]);
  });

  it("computes symmetric normalised Laplacian with expected properties", () => {
    const A = tf.tensor2d(
      [
        [0, 1],
        [1, 0],
      ],
      [2, 2],
    );
    const L = normalised_laplacian(A);
    const Larr = L.arraySync() as number[][];
    // For a 2-node line the normalised Laplacian is [[1,-1],[ -1,1 ]]
    // after degree scaling (degrees are both 1 ⇒ scaling leaves unchanged)
    expect(Larr[0][0]).toBeCloseTo(1);
    expect(Larr[0][1]).toBeCloseTo(-1);
    expect(Larr[1][0]).toBeCloseTo(-1);
    expect(Larr[1][1]).toBeCloseTo(1);
  });

  it("performs eigen decomposition and returns smallest eigenvectors", () => {
    // Simple 2-node graph Laplacian from previous test
    const L = tf.tensor2d(
      [
        [1, -1],
        [-1, 1],
      ],
      [2, 2],
    );

    const { eigenvalues } = jacobi_eigen_decomposition(L);
    // Expected eigenvalues: 0 (multiplicity 1), 2 (multiplicity 1)
    expect(eigenvalues[0]).toBeCloseTo(0, 6);
    expect(eigenvalues[1]).toBeCloseTo(2, 6);

    // Smallest eigenvector (k=1) corresponds to [1/sqrt(2), 1/sqrt(2)] or its negation
    const vecs = smallest_eigenvectors(L, 1).arraySync() as number[][];
    const v0 = vecs.map((row) => row[0]);
    const norm = Math.hypot(...v0);
    // Normalise for comparison
    const v0n = v0.map((x) => x / norm);
    expect(Math.abs(v0n[0])).toBeCloseTo(Math.SQRT1_2, 3);
    expect(Math.abs(v0n[1])).toBeCloseTo(Math.SQRT1_2, 3);
  });
});

