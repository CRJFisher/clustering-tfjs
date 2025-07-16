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

    // Expect symmetric binary matrix with 2*(n-1) edges (since chain). For 4 nodes and k=1 we have 6 ones excluding diag.
    let edgeCount = 0;
    for (let i = 0; i < 4; i++) {
      expect(arr[i][i]).toBe(0);
      for (let j = 0; j < 4; j++) {
        expect(arr[i][j]).toBe(arr[j][i]);
        if (i !== j && arr[i][j] === 1) edgeCount++;
      }
    }
    expect(edgeCount).toBe(6);
  });
});

