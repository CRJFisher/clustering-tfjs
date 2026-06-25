import * as tf from "../../test_support/tensorflow_helper";

import {
  compute_rbf_affinity,
  compute_knn_affinity,
  compute_sparse_knn_affinity,
  compute_cosine_affinity,
} from "./affinity";
import { sparse_to_dense_array } from "./sparse";

describe("compute_rbf_affinity", () => {
  it("produces a symmetric matrix with ones on the diagonal", () => {
    const X = tf.tensor2d([
      [0, 0],
      [1, 0],
      [0, 1],
    ]);
    const A = compute_rbf_affinity(X, 1);
    expect(A.shape).toEqual([3, 3]);
    const arr = A.arraySync() as number[][];
    A.dispose();
    X.dispose();
    for (let i = 0; i < 3; i++) {
      expect(arr[i][i]).toBeCloseTo(1);
      for (let j = 0; j < 3; j++) {
        expect(arr[i][j]).toBeCloseTo(arr[j][i]);
      }
    }
  });

  it("off-diagonal value equals exp(-gamma * squared_distance)", () => {
    const X = tf.tensor2d([[0, 0], [1, 0]]);
    const At = compute_rbf_affinity(X, 1);
    const arr = At.arraySync() as number[][];
    At.dispose();
    X.dispose();
    expect(arr[0][1]).toBeCloseTo(Math.exp(-1), 6);
  });

  it("defaults gamma to 1 / n_features when unspecified", () => {
    // n_features = 2 → gamma = 0.5; squared distance 1 → exp(-0.5).
    const X = tf.tensor2d([[0, 0], [1, 0]]);
    const At = compute_rbf_affinity(X);
    const arr = At.arraySync() as number[][];
    At.dispose();
    X.dispose();
    expect(arr[0][1]).toBeCloseTo(Math.exp(-0.5), 6);
  });
});

describe("compute_knn_affinity", () => {
  it("produces correct edge count and symmetric weights", () => {
    const X = tf.tensor2d([[0], [1], [2], [3]]);
    const A = compute_knn_affinity(X, 1);
    const arr = A.arraySync() as number[][];
    A.dispose();
    X.dispose();
    let edge_count = 0;
    for (let i = 0; i < 4; i++) {
      expect(arr[i][i]).toBe(1);
      for (let j = 0; j < 4; j++) {
        expect(arr[i][j]).toBe(arr[j][i]);
        if (arr[i][j] > 0) edge_count++;
      }
    }
    expect(edge_count).toBe(4);
  });

  it("include_self=false yields zero diagonal and preserves symmetry", () => {
    const X = tf.tensor2d([[0], [1], [2], [3]]);
    const At = compute_knn_affinity(X, 1, false);
    const arr = At.arraySync() as number[][];
    At.dispose();
    X.dispose();
    for (let i = 0; i < 4; i++) {
      expect(arr[i][i]).toBe(0);
      for (let j = 0; j < 4; j++) {
        expect(arr[i][j]).toBe(arr[j][i]);
      }
    }
  });
});

describe("compute_sparse_knn_affinity", () => {
  const X = () => tf.tensor2d([[0], [1], [2], [3]]);

  it("matches the dense compute_knn_affinity output", () => {
    const points = tf.tensor2d([[0], [1], [2], [4]]);

    const dense = compute_knn_affinity(points, 2);
    const sparse = compute_sparse_knn_affinity(points, 2);

    expect(sparse.rows).toBe(4);
    expect(sparse.cols).toBe(4);
    expect(sparse_to_dense_array(sparse)).toEqual(dense.arraySync());

    dense.dispose();
    points.dispose();
  });

  it("rejects a non-positive or non-integer k", () => {
    const points = X();
    expect(() => compute_sparse_knn_affinity(points, 0)).toThrow(
      "positive integer",
    );
    expect(() => compute_sparse_knn_affinity(points, 1.5)).toThrow(
      "positive integer",
    );
    points.dispose();
  });

  it("rejects k >= number of samples", () => {
    const points = X();
    expect(() => compute_sparse_knn_affinity(points, 4)).toThrow(
      "smaller than the number of samples",
    );
    points.dispose();
  });

  it("rejects an empty point set", () => {
    const empty = tf.tensor2d([], [0, 2]);
    expect(() => compute_sparse_knn_affinity(empty, 1)).toThrow(
      "at least one sample",
    );
    empty.dispose();
  });
});

describe("compute_cosine_affinity", () => {
  it("produces a symmetric similarity matrix with unit diagonal", () => {
    const X = tf.tensor2d([
      [1, 0],
      [0, 1],
      [1, 1],
    ]);
    const A = compute_cosine_affinity(X);
    const arr = A.arraySync() as number[][];

    for (let i = 0; i < 3; i++) expect(arr[i][i]).toBeCloseTo(1, 6);
    for (let i = 0; i < 3; i++)
      for (let j = 0; j < 3; j++) expect(arr[i][j]).toBeCloseTo(arr[j][i], 6);
    // cos([1,0], [0,1]) = 0 → affinity 0
    expect(arr[0][1]).toBeCloseTo(0, 6);
    // cos([1,0], [1,1]) = 1/√2
    expect(arr[0][2]).toBeCloseTo(1 / Math.sqrt(2), 5);

    A.dispose();
    X.dispose();
  });
});

describe("compute_knn_affinity – tensor lifecycle", () => {
  const run_once = (): void => {
    const X = tf.tensor2d([
      [0, 0],
      [1, 0],
      [0, 1],
      [1, 1],
      [5, 5],
    ]);
    const affinity = compute_knn_affinity(X, 2, true);
    affinity.dispose();
    if (!X.isDisposed) X.dispose();
  };

  it("does not leak tensors across repeated calls", () => {
    run_once(); // warm up one-time allocations
    const before = tf.memory().numTensors;
    for (let i = 0; i < 5; i++) run_once();
    expect(tf.memory().numTensors).toBe(before);
  });
});
