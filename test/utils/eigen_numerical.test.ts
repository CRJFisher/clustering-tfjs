import * as tf from "@tensorflow/tfjs-node";

import {
  normalised_laplacian,
  jacobi_eigen_decomposition,
} from "../../src";

/**
 * Unit tests covering numerical-stability edge cases for the Jacobi eigen
 * decomposition helper.
 */

describe("jacobi_eigen_decomposition – numerical edge cases", () => {
  it("handles disconnected graphs (block-diagonal Laplacian)", () => {
    /*
     * Two disconnected components, each a 2-node line.  The affinity
     * matrix is block-diagonal which results in a *normalised* Laplacian
     * that is also block-diagonal.  We therefore expect **two** zero
     * eigenvalues corresponding to the number of connected components.
     */
    const A = tf.tensor2d(
      [
        // component 1
        [0, 1, 0, 0],
        [1, 0, 0, 0],
        // component 2
        [0, 0, 0, 1],
        [0, 0, 1, 0],
      ],
      [4, 4],
    );

    const L = normalised_laplacian(A);
    const { eigenvalues } = jacobi_eigen_decomposition(L);

    // Sort ascending just in case
    const sorted = [...eigenvalues].sort((a, b) => a - b);
    // First two eigenvalues should be ~0 (within tolerance)
    expect(Math.abs(sorted[0])).toBeLessThan(1e-8);
    expect(Math.abs(sorted[1])).toBeLessThan(1e-8);
  });

  it("does not produce NaN/Inf for nearly-identical points", () => {
    /*
     * Create an affinity matrix for two almost identical points with a very
     * small perturbation which leads to an *ill-conditioned* Laplacian.  The
     * decomposition should still return finite numbers.
     */
    const eps = 1e-12;
    const A = tf.tensor2d(
      [
        [0, 1 + eps],
        [1 + eps, 0],
      ],
      [2, 2],
    );

    const L = normalised_laplacian(A);
    const { eigenvalues, eigenvectors } = jacobi_eigen_decomposition(L);

    // All eigenvalues must be finite numbers (no NaN, Inf)
    eigenvalues.forEach((v) => {
      expect(Number.isFinite(v)).toBe(true);
    });

    // Eigenvectors entries must also be finite
    eigenvectors.flat().forEach((v) => {
      expect(Number.isFinite(v)).toBe(true);
    });
  });
});

