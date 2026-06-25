import * as tf from "../../test_support/tensorflow_helper";

import {
  degree_vector,
  normalised_laplacian,
  sparse_normalised_laplacian_operator,
  jacobi_eigen_decomposition,
  smallest_eigenvectors,
} from "./laplacian";
import { sparse_matrix_from_row_maps } from "./sparse";

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

  it("rejects a non-square affinity in degree_vector", () => {
    const A = tf.tensor2d([[1, 2, 3]], [1, 3]);
    expect(() => degree_vector(A)).toThrow("square");
    A.dispose();
  });

  it("rejects a non-positive or non-integer k in smallest_eigenvectors", () => {
    const L = tf.tensor2d([
      [1, -1],
      [-1, 1],
    ]);
    expect(() => smallest_eigenvectors(L, 0)).toThrow("positive integer");
    expect(() => smallest_eigenvectors(L, 1.5)).toThrow("positive integer");
    L.dispose();
  });

  it("return_diag=true also yields D^{-1/2} per vertex", () => {
    // Degrees are 1 and 4, so D^{-1/2} = [1, 1/2].
    const A = tf.tensor2d([
      [0, 4],
      [4, 0],
    ]);
    const { laplacian, sqrt_degrees } = normalised_laplacian(A, true);
    const diag = Array.from(sqrt_degrees.dataSync());
    expect(diag[0]).toBeCloseTo(1 / Math.sqrt(4), 6);
    expect(diag[1]).toBeCloseTo(1 / Math.sqrt(4), 6);
    // Off-diagonal of the normalised Laplacian is -A/√(d_i d_j) = -4/4 = -1.
    const Larr = laplacian.arraySync() as number[][];
    expect(Larr[0][1]).toBeCloseTo(-1, 6);
    laplacian.dispose();
    sqrt_degrees.dispose();
    A.dispose();
  });

  it("treats an isolated vertex as an identity row/column", () => {
    // Vertex 2 has degree 0; its Laplacian row must be the identity row.
    const A = tf.tensor2d([
      [0, 1, 0],
      [1, 0, 0],
      [0, 0, 0],
    ]);
    const L = normalised_laplacian(A).arraySync() as number[][];
    expect(L[2]).toEqual([0, 0, 1]);
    expect(L[0][2]).toBe(0);
    expect(L[1][2]).toBe(0);
  });
});

describe("sparse_normalised_laplacian_operator", () => {
  // Symmetric 3-node affinity with zero diagonal.
  const build_affinity = () =>
    sparse_matrix_from_row_maps([
      new Map([
        [1, 1],
        [2, 2],
      ]),
      new Map([
        [0, 1],
        [2, 3],
      ]),
      new Map([
        [0, 2],
        [1, 3],
      ]),
    ]);

  it("matvec matches the dense normalised Laplacian applied to a vector", () => {
    const sparse = build_affinity();
    const { operator } = sparse_normalised_laplacian_operator(sparse);

    const v = new Float64Array([1, 2, 3]);
    const got = operator.matvec(v);

    // Reference: dense L · v.
    const dense_affinity = tf.tensor2d([
      [0, 1, 2],
      [1, 0, 3],
      [2, 3, 0],
    ]);
    const L = normalised_laplacian(dense_affinity);
    const expected = Array.from(
      (L.matMul(tf.tensor2d([[1], [2], [3]])).dataSync()),
    );
    L.dispose();
    dense_affinity.dispose();

    // The dense reference runs on the float32 tfjs backend while the sparse
    // operator is float64, so parity holds to ~1e-7, not machine precision.
    for (let i = 0; i < 3; i++) {
      expect(got[i]).toBeCloseTo(expected[i], 6);
    }
  });

  it("exposes degrees and D^{-1/2} consistent with the affinity", () => {
    const sparse = build_affinity();
    const { degrees, sqrt_degrees } = sparse_normalised_laplacian_operator(sparse);
    expect(Array.from(degrees)).toEqual([3, 4, 5]);
    expect(sqrt_degrees[0]).toBeCloseTo(Math.pow(3, -0.5), 12);
    expect(sqrt_degrees[2]).toBeCloseTo(Math.pow(5, -0.5), 12);
  });

  it("rejects a non-square affinity", () => {
    const rect = sparse_matrix_from_row_maps([new Map([[0, 1]])], 5);
    expect(() => sparse_normalised_laplacian_operator(rect)).toThrow("square");
  });

  it("rejects a vector whose length does not match the operator size", () => {
    const { operator } = sparse_normalised_laplacian_operator(build_affinity());
    expect(() => operator.matvec(new Float64Array([1, 2]))).toThrow(
      "does not match Laplacian size",
    );
  });
});

