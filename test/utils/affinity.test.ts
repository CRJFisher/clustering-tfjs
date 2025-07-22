import * as tf from "@tensorflow/tfjs-node";

import {
  compute_rbf_affinity,
  compute_knn_affinity,
} from "../../src/utils/affinity";

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
    // When includeSelf=true (default), diagonal has 1s
    // When symmetrized, edges that appear in only one direction get 0.5
    let edgeCount = 0;
    for (let i = 0; i < 4; i++) {
      expect(arr[i][i]).toBe(1); // Self-loops included by default
      for (let j = 0; j < 4; j++) {
        expect(arr[i][j]).toBe(arr[j][i]);
        if (arr[i][j] > 0) edgeCount++;
      }
    }
    // With k=1 and includeSelf=true (default), sklearn behavior means k=1 includes self
    // So each node only connects to its 1 nearest neighbor (which is itself)
    // Result: diagonal matrix with 4 edges
    expect(edgeCount).toBe(4);
  });
});

