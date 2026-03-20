import { describe, it, expect } from "@jest/globals";
import * as tf from "../tensorflow-helper";
import { adjustedRandIndex } from "../../src/validation/adjusted_rand_index";

describe("Adjusted Rand Index", () => {
  describe("Basic functionality", () => {
    it("should return 1.0 for perfect agreement", () => {
      const labelsTrue = [0, 0, 1, 1, 2, 2];
      const labelsPred = [0, 0, 1, 1, 2, 2];

      expect(adjustedRandIndex(labelsTrue, labelsPred)).toBeCloseTo(1.0, 10);
    });

    it("should be permutation invariant", () => {
      const labelsTrue = [0, 0, 1, 1];
      const labelsPred = [1, 1, 0, 0];

      expect(adjustedRandIndex(labelsTrue, labelsPred)).toBeCloseTo(1.0, 10);
    });

    it("should return near 0 for random labeling", () => {
      // Large enough to be stable
      const labelsTrue = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1];
      const labelsPred = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1];

      const ari = adjustedRandIndex(labelsTrue, labelsPred);
      expect(Math.abs(ari)).toBeLessThan(0.5);
    });

    it("should return negative for worse-than-random", () => {
      // sklearn: adjusted_rand_score([0,0,1,1], [0,1,0,1]) = -0.5
      const labelsTrue = [0, 0, 1, 1];
      const labelsPred = [0, 1, 0, 1];

      const ari = adjustedRandIndex(labelsTrue, labelsPred);
      expect(ari).toBeCloseTo(-0.5, 5);
    });
  });

  describe("sklearn reference values", () => {
    it("should match sklearn for partial agreement", () => {
      // sklearn: adjusted_rand_score([0,0,1,2], [0,0,1,1]) ≈ 0.5714
      const labelsTrue = [0, 0, 1, 2];
      const labelsPred = [0, 0, 1, 1];

      const ari = adjustedRandIndex(labelsTrue, labelsPred);
      expect(ari).toBeCloseTo(0.5714, 3);
    });
  });

  describe("Edge cases", () => {
    it("should return 1.0 for all-same labels in both (identical partitions)", () => {
      const labelsTrue = [0, 0, 0];
      const labelsPred = [0, 0, 0];

      expect(adjustedRandIndex(labelsTrue, labelsPred)).toBeCloseTo(1.0, 10);
    });

    it("should return 1.0 for each-point-own-cluster in both (identical partitions)", () => {
      // When every point is a singleton in both, the partitions are identical
      const labelsTrue = [0, 1, 2, 3];
      const labelsPred = [4, 5, 6, 7];

      expect(adjustedRandIndex(labelsTrue, labelsPred)).toBeCloseTo(1.0, 10);
    });

    it("should throw for empty arrays", () => {
      expect(() => adjustedRandIndex([], [])).toThrow("must not be empty");
    });

    it("should throw for mismatched lengths", () => {
      expect(() => adjustedRandIndex([0, 1], [0, 1, 2])).toThrow(
        "same length"
      );
    });

    it("should return 1.0 for single sample (trivially identical)", () => {
      expect(adjustedRandIndex([0], [0])).toBeCloseTo(1.0, 10);
    });
  });

  describe("Tensor inputs", () => {
    it("should produce same result with tensor inputs", () => {
      const labelsTrue = [0, 0, 1, 1, 2, 2];
      const labelsPred = [0, 0, 1, 1, 2, 2];

      const arrayResult = adjustedRandIndex(labelsTrue, labelsPred);

      const trueTensor = tf.tensor1d(labelsTrue);
      const predTensor = tf.tensor1d(labelsPred);
      const tensorResult = adjustedRandIndex(trueTensor, predTensor);

      expect(tensorResult).toBeCloseTo(arrayResult, 10);

      trueTensor.dispose();
      predTensor.dispose();
    });
  });
});
