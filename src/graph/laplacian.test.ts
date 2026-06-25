import * as tf from "../../test_support/tensorflow_helper";

import {
  degree_vector,
  normalised_laplacian,
  sparse_normalised_laplacian_operator,
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

  it("rejects a non-square affinity in degree_vector", () => {
    const A = tf.tensor2d([[1, 2, 3]], [1, 3]);
    expect(() => degree_vector(A)).toThrow("square");
    A.dispose();
  });

  it("return_diag=true also yields D^{-1/2} per vertex", () => {
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

  it("normalised_laplacian is symmetric for a symmetric affinity matrix", () => {
    const A = tf.tensor2d([
      [0, 2, 1],
      [2, 0, 3],
      [1, 3, 0],
    ]);
    const L = normalised_laplacian(A).arraySync() as number[][];
    for (let i = 0; i < 3; i++) {
      for (let j = 0; j < 3; j++) {
        expect(L[i][j]).toBeCloseTo(L[j][i], 6);
      }
    }
  });

  it("degree_vector sums rows of a non-square-shaped input", () => {
    const A_rect = tf.tensor2d([[1, 2, 3]], [1, 3]);
    expect(() => degree_vector(A_rect)).toThrow("square");
    A_rect.dispose();
  });
});

describe("sparse_normalised_laplacian_operator", () => {
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

  it("isolated vertex (degree 0) gets inv_sqrt_degree = 1 and matvec row = v[i]", () => {
    // Node 2 is isolated: matvec must return v[2] unchanged (identity row).
    const sparse = sparse_matrix_from_row_maps([
      new Map([[1, 1]]),
      new Map([[0, 1]]),
      new Map(),
    ]);
    const { operator, degrees, sqrt_degrees } = sparse_normalised_laplacian_operator(sparse);
    expect(degrees[2]).toBe(0);
    expect(sqrt_degrees[2]).toBe(1);
    const v = new Float64Array([0, 0, 7]);
    const result = operator.matvec(v);
    expect(result[2]).toBeCloseTo(7, 10);
  });
});

