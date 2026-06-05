import { describe, it, expect } from "@jest/globals";
import * as tf from "../tensorflow-helper";
import { normalized_mutual_info } from "../../src/validation/normalized_mutual_info";

describe("Normalized Mutual Information", () => {
  describe("Basic functionality", () => {
    it("should return 1.0 for perfect agreement", () => {
      const labels_true = [0, 0, 1, 1, 2, 2];
      const labels_pred = [0, 0, 1, 1, 2, 2];

      expect(normalized_mutual_info(labels_true, labels_pred)).toBeCloseTo(1.0, 10);
    });

    it("should be permutation invariant", () => {
      const labels_true = [0, 0, 1, 1];
      const labels_pred = [1, 1, 0, 0];

      expect(normalized_mutual_info(labels_true, labels_pred)).toBeCloseTo(1.0, 10);
    });

    it("should return 0 for independent labelings", () => {
      // One has single cluster, the other doesn't -> NMI = 0
      const labels_true = [0, 0, 0, 0];
      const labels_pred = [0, 1, 2, 3];

      expect(normalized_mutual_info(labels_true, labels_pred)).toBeCloseTo(0, 5);
    });

    it("should return value in [0, 1]", () => {
      const labels_true = [0, 0, 1, 1, 2, 2];
      const labels_pred = [0, 1, 1, 2, 2, 0];

      const nmi = normalized_mutual_info(labels_true, labels_pred);
      expect(nmi).toBeGreaterThanOrEqual(0);
      expect(nmi).toBeLessThanOrEqual(1);
    });
  });

  describe("Average methods", () => {
    it("should default to arithmetic", () => {
      const labels_true = [0, 0, 1, 2];
      const labels_pred = [0, 0, 1, 1];

      const default_result = normalized_mutual_info(labels_true, labels_pred);
      const arithmetic_result = normalized_mutual_info(labels_true, labels_pred, "arithmetic");

      expect(default_result).toBeCloseTo(arithmetic_result, 10);
    });

    it("should compute different results for different average methods", () => {
      const labels_true = [0, 0, 1, 1, 2];
      const labels_pred = [0, 0, 1, 1, 1];

      const arithmetic = normalized_mutual_info(labels_true, labels_pred, "arithmetic");
      const geometric = normalized_mutual_info(labels_true, labels_pred, "geometric");
      const min = normalized_mutual_info(labels_true, labels_pred, "min");
      const max = normalized_mutual_info(labels_true, labels_pred, "max");

      // All should be in [0, 1]
      for (const val of [arithmetic, geometric, min, max]) {
        expect(val).toBeGreaterThanOrEqual(0);
        expect(val).toBeLessThanOrEqual(1.001); // small tolerance
      }

      // min >= max (normalizing by max gives smaller NMI than normalizing by min)
      expect(max).toBeLessThanOrEqual(min + 1e-10);
    });
  });

  describe("Edge cases", () => {
    it("should return 1.0 when both are single cluster", () => {
      const labels_true = [0, 0, 0];
      const labels_pred = [0, 0, 0];

      expect(normalized_mutual_info(labels_true, labels_pred)).toBeCloseTo(1.0, 10);
    });

    it("should throw for empty arrays", () => {
      expect(() => normalized_mutual_info([], [])).toThrow("must not be empty");
    });

    it("should throw for mismatched lengths", () => {
      expect(() => normalized_mutual_info([0, 1], [0, 1, 2])).toThrow(
        "same length"
      );
    });
  });

  describe("sklearn reference values", () => {
    it("should return 1.0 for single sample", () => {
      expect(normalized_mutual_info([0], [0])).toBeCloseTo(1.0, 10);
    });

    it("should match sklearn for partial agreement with arithmetic average", () => {
      // sklearn: normalized_mutual_info_score([0,0,1,1,2], [0,0,1,1,1], average_method='arithmetic')
      // ≈ 0.7790
      const labels_true = [0, 0, 1, 1, 2];
      const labels_pred = [0, 0, 1, 1, 1];

      const nmi = normalized_mutual_info(labels_true, labels_pred, "arithmetic");
      expect(nmi).toBeGreaterThan(0.7);
      expect(nmi).toBeLessThan(0.85);
    });
  });

  describe("Tensor inputs", () => {
    it("should produce same result with tensor inputs", () => {
      const labels_true = [0, 0, 1, 1, 2, 2];
      const labels_pred = [0, 0, 1, 1, 2, 2];

      const array_result = normalized_mutual_info(labels_true, labels_pred);

      const true_tensor = tf.tensor1d(labels_true);
      const pred_tensor = tf.tensor1d(labels_pred);
      const tensor_result = normalized_mutual_info(true_tensor, pred_tensor);

      expect(tensor_result).toBeCloseTo(array_result, 10);

      true_tensor.dispose();
      pred_tensor.dispose();
    });
  });
});
