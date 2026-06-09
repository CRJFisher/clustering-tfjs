import * as tf from "../../test_support/tensorflow_helper";

import {
  compute_rbf_affinity,
  compute_knn_affinity,
  compute_sparse_knn_affinity,
  compute_cosine_affinity,
  compute_affinity_matrix,
} from "./affinity";
import { sparse_to_dense_array } from "./sparse";

describe("Affinity matrix utilities", () => {
  afterEach(() => tf.engine().disposeVariables());

  it("computes symmetric RBF affinity with ones on diagonal", () => {
    const X = tf.tensor2d([
      [0, 0],
      [1, 0],
      [0, 1],
    ]);

    const A = compute_rbf_affinity(X, 1);

    expect(A.shape).toEqual([3, 3]);

    const arr = A.arraySync() as number[][];
    for (let i = 0; i < 3; i++) {
      expect(arr[i][i]).toBeCloseTo(1);
      for (let j = 0; j < 3; j++) {
        expect(arr[i][j]).toBeCloseTo(arr[j][i]);
      }
    }
  });

  it("computes k-NN affinity with correct number of edges and symmetrisation", () => {
    // Four points on a line at positions 0,1,2,3
    const X = tf.tensor2d([[0], [1], [2], [3]]);

    const A = compute_knn_affinity(X, 1); // connect each point to its nearest neighbour

    const arr = A.arraySync() as number[][];

    // Expect symmetric matrix with values 0, 0.5, or 1
    // When include_self=true (default), diagonal has 1s
    // When symmetrized, edges that appear in only one direction get 0.5
    let edge_count = 0;
    for (let i = 0; i < 4; i++) {
      expect(arr[i][i]).toBe(1); // Self-loops included by default
      for (let j = 0; j < 4; j++) {
        expect(arr[i][j]).toBe(arr[j][i]);
        if (arr[i][j] > 0) edge_count++;
      }
    }
    // With k=1 and include_self=true (default), sklearn behavior means k=1 includes self
    // So each node only connects to its 1 nearest neighbor (which is itself)
    // Result: diagonal matrix with 4 edges
    expect(edge_count).toBe(4);
  });

  it("computes sparse k-NN affinity matching the dense helper", () => {
    const X = tf.tensor2d([[0], [1], [2], [4]]);

    const dense = compute_knn_affinity(X, 2);
    const sparse = compute_sparse_knn_affinity(X, 2);

    expect(sparse.rows).toBe(4);
    expect(sparse.cols).toBe(4);
    expect(sparse_to_dense_array(sparse)).toEqual(dense.arraySync());

    dense.dispose();
    X.dispose();
  });
});


describe("compute_cosine_affinity", () => {
  afterEach(() => tf.engine().disposeVariables());

  it("produces a symmetric similarity matrix with unit diagonal", () => {
    const X = tf.tensor2d([
      [1, 0],
      [0, 1],
      [1, 1],
    ]);
    const A = compute_cosine_affinity(X);
    const arr = A.arraySync() as number[][];

    // Diagonal forced to 1.
    for (let i = 0; i < 3; i++) expect(arr[i][i]).toBeCloseTo(1, 6);
    // Symmetric.
    for (let i = 0; i < 3; i++)
      for (let j = 0; j < 3; j++) expect(arr[i][j]).toBeCloseTo(arr[j][i], 6);
    // cos similarity between [1,0] and [0,1] is 0 -> affinity 0.
    expect(arr[0][1]).toBeCloseTo(0, 6);
    // cos similarity between [1,0] and [1,1] is 1/sqrt(2).
    expect(arr[0][2]).toBeCloseTo(1 / Math.sqrt(2), 5);

    A.dispose();
    X.dispose();
  });

  it("is reachable through compute_affinity_matrix dispatcher", () => {
    const X = tf.tensor2d([
      [1, 0],
      [0, 1],
    ]);
    const A = compute_affinity_matrix(X, { affinity: "cosine" });
    const arr = A.arraySync() as number[][];
    expect(arr[0][0]).toBeCloseTo(1, 6);
    expect(arr[0][1]).toBeCloseTo(0, 6);
    A.dispose();
    X.dispose();
  });
});

describe("compute_knn_affinity – tensor lifecycle (AC 49.1 #7)", () => {
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
