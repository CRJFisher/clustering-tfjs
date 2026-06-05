import { describe, it, expect } from "@jest/globals";
import * as tf from "../tensorflow-helper";
import { adjusted_rand_index } from "../../src/validation/adjusted_rand_index";

describe("Adjusted Rand Index", () => {
  describe("Basic functionality", () => {
    it("should return 1.0 for perfect agreement", () => {
      const labels_true = [0, 0, 1, 1, 2, 2];
      const labels_pred = [0, 0, 1, 1, 2, 2];

      expect(adjusted_rand_index(labels_true, labels_pred)).toBeCloseTo(1.0, 10);
    });

    it("should be permutation invariant", () => {
      const labels_true = [0, 0, 1, 1];
      const labels_pred = [1, 1, 0, 0];

      expect(adjusted_rand_index(labels_true, labels_pred)).toBeCloseTo(1.0, 10);
    });

    it("should return near 0 for random labeling", () => {
      // Large enough to be stable
      const labels_true = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1];
      const labels_pred = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1];

      const ari = adjusted_rand_index(labels_true, labels_pred);
      expect(Math.abs(ari)).toBeLessThan(0.5);
    });

    it("should return negative for worse-than-random", () => {
      // sklearn: adjusted_rand_score([0,0,1,1], [0,1,0,1]) = -0.5
      const labels_true = [0, 0, 1, 1];
      const labels_pred = [0, 1, 0, 1];

      const ari = adjusted_rand_index(labels_true, labels_pred);
      expect(ari).toBeCloseTo(-0.5, 5);
    });
  });

  describe("sklearn reference values", () => {
    it("should match sklearn for partial agreement", () => {
      // sklearn: adjusted_rand_score([0,0,1,2], [0,0,1,1]) ≈ 0.5714
      const labels_true = [0, 0, 1, 2];
      const labels_pred = [0, 0, 1, 1];

      const ari = adjusted_rand_index(labels_true, labels_pred);
      expect(ari).toBeCloseTo(0.5714, 3);
    });
  });

  describe("Edge cases", () => {
    it("should return 1.0 for all-same labels in both (identical partitions)", () => {
      const labels_true = [0, 0, 0];
      const labels_pred = [0, 0, 0];

      expect(adjusted_rand_index(labels_true, labels_pred)).toBeCloseTo(1.0, 10);
    });

    it("should return 1.0 for each-point-own-cluster in both (identical partitions)", () => {
      // When every point is a singleton in both, the partitions are identical
      const labels_true = [0, 1, 2, 3];
      const labels_pred = [4, 5, 6, 7];

      expect(adjusted_rand_index(labels_true, labels_pred)).toBeCloseTo(1.0, 10);
    });

    it("should throw for empty arrays", () => {
      expect(() => adjusted_rand_index([], [])).toThrow("must not be empty");
    });

    it("should throw for mismatched lengths", () => {
      expect(() => adjusted_rand_index([0, 1], [0, 1, 2])).toThrow(
        "same length"
      );
    });

    it("should return 1.0 for single sample (trivially identical)", () => {
      expect(adjusted_rand_index([0], [0])).toBeCloseTo(1.0, 10);
    });
  });

  describe("Tensor inputs", () => {
    it("should produce same result with tensor inputs", () => {
      const labels_true = [0, 0, 1, 1, 2, 2];
      const labels_pred = [0, 0, 1, 1, 2, 2];

      const array_result = adjusted_rand_index(labels_true, labels_pred);

      const true_tensor = tf.tensor1d(labels_true);
      const pred_tensor = tf.tensor1d(labels_pred);
      const tensor_result = adjusted_rand_index(true_tensor, pred_tensor);

      expect(tensor_result).toBeCloseTo(array_result, 10);

      true_tensor.dispose();
      pred_tensor.dispose();
    });
  });
});
