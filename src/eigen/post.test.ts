import * as tf from "../../test_support/tensorflow_helper";

import { deterministic_eigenpair_processing } from "./post";
import { improved_jacobi_eigen } from "./improved";

describe("deterministic_eigenpair_processing – integration", () => {
  it("sorts eigenpairs ascending and applies sign flip", () => {
    const M = tf.tensor2d(
      [
        [2, 1, 0],
        [1, 2, 0],
        [0, 0, 3],
      ],
      [3, 3],
      "float32",
    );

    const { eigenvalues, eigenvectors } = improved_jacobi_eigen(M as tf.Tensor2D);

    const processed = deterministic_eigenpair_processing({
      eigenvalues,
      eigenvectors,
    });

    const vals = processed.eigenvalues;
    expect(vals.length).toBe(3);
    expect(vals[0]).toBeCloseTo(1, 5);
    expect(vals[1]).toBeCloseTo(3, 5);
    expect(vals[2]).toBeCloseTo(3, 5);

    const vecs = processed.eigenvectors;
    for (let col = 0; col < 3; col++) {
      let max_abs = 0;
      let max_idx = 0;
      for (let row = 0; row < 3; row++) {
        const abs = Math.abs(vecs[row][col]);
        if (abs > max_abs) {
          max_abs = abs;
          max_idx = row;
        }
      }
      expect(vecs[max_idx][col]).toBeGreaterThanOrEqual(0);
    }

    M.dispose();
  });
});

describe("deterministic_eigenpair_processing – direct contract", () => {
  it("reorders columns by ascending eigenvalue and flips signs to a positive max-abs entry", () => {
    // Columns (by eigenvalue): 3 -> [0.6,-0.8], 1 -> [0.3,0.4], 2 -> [-0.9,0.1].
    const { eigenvalues, eigenvectors } = deterministic_eigenpair_processing({
      eigenvalues: [3, 1, 2],
      eigenvectors: [
        [0.6, 0.3, -0.9],
        [-0.8, 0.4, 0.1],
      ],
    });

    expect(eigenvalues).toEqual([1, 2, 3]);
    // col 1 (max-abs 0.4 already positive) stays; col 2 (max-abs -0.9) flips;
    // col 3 (max-abs -0.8) flips.
    expect(eigenvectors[0][0]).toBeCloseTo(0.3, 10);
    expect(eigenvectors[1][0]).toBeCloseTo(0.4, 10);
    expect(eigenvectors[0][1]).toBeCloseTo(0.9, 10);
    expect(eigenvectors[1][1]).toBeCloseTo(-0.1, 10);
    expect(eigenvectors[0][2]).toBeCloseTo(-0.6, 10);
    expect(eigenvectors[1][2]).toBeCloseTo(0.8, 10);
  });

  it("is a no-op when eigenvalues are already in ascending order", () => {
    const input = {
      eigenvalues: [1, 2, 3],
      eigenvectors: [
        [0.5, 0.3, 0.1],
        [0.5, 0.3, 0.1],
      ],
    };
    const { eigenvalues, eigenvectors } = deterministic_eigenpair_processing(input);
    expect(eigenvalues).toEqual([1, 2, 3]);
    expect(eigenvectors[0][0]).toBeCloseTo(0.5, 10);
    expect(eigenvectors[0][1]).toBeCloseTo(0.3, 10);
    expect(eigenvectors[0][2]).toBeCloseTo(0.1, 10);
  });

  it("supports fewer eigenpairs than rows (m < n)", () => {
    // 3 rows, 2 columns: eigenvalues 5 -> [0.1,0.2,-0.95], 1 -> [0.7,0.1,0.2].
    const { eigenvalues, eigenvectors } = deterministic_eigenpair_processing({
      eigenvalues: [5, 1],
      eigenvectors: [
        [0.1, 0.7],
        [0.2, 0.1],
        [-0.95, 0.2],
      ],
    });

    expect(eigenvalues).toEqual([1, 5]);
    expect(eigenvectors.length).toBe(3);
    expect(eigenvectors[0].length).toBe(2);
    // col for eigenvalue 1 stays (max-abs 0.7 positive); col for 5 flips.
    expect(eigenvectors[2][0]).toBeCloseTo(0.2, 10);
    expect(eigenvectors[2][1]).toBeCloseTo(0.95, 10);
  });

  it("handles an all-zero column without crashing", () => {
    // max_abs stays 0, sign defaults to +1 (0 is not < 0), output is all-zero.
    const { eigenvalues, eigenvectors } = deterministic_eigenpair_processing({
      eigenvalues: [1, 2],
      eigenvectors: [
        [0, 0.5],
        [0, -0.5],
      ],
    });
    expect(eigenvalues).toEqual([1, 2]);
    expect(eigenvectors[0][0]).toBe(0);
    expect(eigenvectors[1][0]).toBe(0);
  });

  it("returns empty structures for empty input", () => {
    const out = deterministic_eigenpair_processing({
      eigenvalues: [],
      eigenvectors: [],
    });
    expect(out.eigenvalues).toEqual([]);
    expect(out.eigenvectors).toEqual([]);
  });

  it("throws when a row's column count does not match the eigenvalue count", () => {
    expect(() =>
      deterministic_eigenpair_processing({
        eigenvalues: [1, 2],
        eigenvectors: [
          [0.1, 0.2],
          [0.3], // wrong width
        ],
      }),
    ).toThrow("column count");
  });
});
