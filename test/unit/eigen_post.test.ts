import * as tf from "@tensorflow/tfjs-node";

import { deterministic_eigenpair_processing } from "../../src/utils/eigen_post";

describe("deterministic_eigenpair_processing", () => {
  it("sorts eigenpairs ascending and applies sign flip", () => {
    // Simple 3×3 symmetric matrix with repeated eigen-values
    // Matrix [[2,1,0],[1,2,0],[0,0,3]] has eigenvalues 1,3,3
    const M = tf.tensor2d(
      [
        [2, 1, 0],
        [1, 2, 0],
        [0, 0, 3],
      ],
      [3, 3],
      "float32",
    );

    const { eigenvalues, eigenvectors } = ((): {
      eigenvalues: number[];
      eigenvectors: number[][];
    } => {
      // tf.linalg.eig is not available; use Jacobi solver from util.
      const { jacobi_eigen_decomposition } = require("../../src/utils/laplacian");
      return jacobi_eigen_decomposition(M as tf.Tensor2D);
    })();

    const processed = deterministic_eigenpair_processing({
      eigenvalues,
      eigenvectors,
    });

    const vals = processed.eigenvalues;
    expect(vals.length).toBe(3);
    // Ascending order – first element should be the smallest eigenvalue (≈1)
    expect(vals[0]).toBeCloseTo(1, 5);
    expect(vals[1]).toBeCloseTo(3, 5);
    expect(vals[2]).toBeCloseTo(3, 5);

    // Check sign convention – max-abs component positive for each vector
    const vecs = processed.eigenvectors;
    for (let col = 0; col < 3; col++) {
      let maxAbs = 0;
      let maxIdx = 0;
      for (let row = 0; row < 3; row++) {
        const abs = Math.abs(vecs[row][col]);
        if (abs > maxAbs) {
          maxAbs = abs;
          maxIdx = row;
        }
      }
      expect(vecs[maxIdx][col]).toBeGreaterThanOrEqual(0);
    }

    M.dispose();
  });
});

