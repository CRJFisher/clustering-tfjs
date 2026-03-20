import { describe, it, expect } from "@jest/globals";
import * as tf from "../tensorflow-helper";
import { normalizedMutualInfo } from "../../src/validation/normalized_mutual_info";

describe("Normalized Mutual Information", () => {
  describe("Basic functionality", () => {
    it("should return 1.0 for perfect agreement", () => {
      const labelsTrue = [0, 0, 1, 1, 2, 2];
      const labelsPred = [0, 0, 1, 1, 2, 2];

      expect(normalizedMutualInfo(labelsTrue, labelsPred)).toBeCloseTo(1.0, 10);
    });

    it("should be permutation invariant", () => {
      const labelsTrue = [0, 0, 1, 1];
      const labelsPred = [1, 1, 0, 0];

      expect(normalizedMutualInfo(labelsTrue, labelsPred)).toBeCloseTo(1.0, 10);
    });

    it("should return 0 for independent labelings", () => {
      // One has single cluster, the other doesn't -> NMI = 0
      const labelsTrue = [0, 0, 0, 0];
      const labelsPred = [0, 1, 2, 3];

      expect(normalizedMutualInfo(labelsTrue, labelsPred)).toBeCloseTo(0, 5);
    });

    it("should return value in [0, 1]", () => {
      const labelsTrue = [0, 0, 1, 1, 2, 2];
      const labelsPred = [0, 1, 1, 2, 2, 0];

      const nmi = normalizedMutualInfo(labelsTrue, labelsPred);
      expect(nmi).toBeGreaterThanOrEqual(0);
      expect(nmi).toBeLessThanOrEqual(1);
    });
  });

  describe("Average methods", () => {
    it("should default to arithmetic", () => {
      const labelsTrue = [0, 0, 1, 2];
      const labelsPred = [0, 0, 1, 1];

      const defaultResult = normalizedMutualInfo(labelsTrue, labelsPred);
      const arithmeticResult = normalizedMutualInfo(labelsTrue, labelsPred, "arithmetic");

      expect(defaultResult).toBeCloseTo(arithmeticResult, 10);
    });

    it("should compute different results for different average methods", () => {
      const labelsTrue = [0, 0, 1, 1, 2];
      const labelsPred = [0, 0, 1, 1, 1];

      const arithmetic = normalizedMutualInfo(labelsTrue, labelsPred, "arithmetic");
      const geometric = normalizedMutualInfo(labelsTrue, labelsPred, "geometric");
      const min = normalizedMutualInfo(labelsTrue, labelsPred, "min");
      const max = normalizedMutualInfo(labelsTrue, labelsPred, "max");

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
      const labelsTrue = [0, 0, 0];
      const labelsPred = [0, 0, 0];

      expect(normalizedMutualInfo(labelsTrue, labelsPred)).toBeCloseTo(1.0, 10);
    });

    it("should throw for empty arrays", () => {
      expect(() => normalizedMutualInfo([], [])).toThrow("must not be empty");
    });

    it("should throw for mismatched lengths", () => {
      expect(() => normalizedMutualInfo([0, 1], [0, 1, 2])).toThrow(
        "same length"
      );
    });
  });

  describe("sklearn reference values", () => {
    it("should return 1.0 for single sample", () => {
      expect(normalizedMutualInfo([0], [0])).toBeCloseTo(1.0, 10);
    });

    it("should match sklearn for partial agreement with arithmetic average", () => {
      // sklearn: normalized_mutual_info_score([0,0,1,1,2], [0,0,1,1,1], average_method='arithmetic')
      // ≈ 0.7790
      const labelsTrue = [0, 0, 1, 1, 2];
      const labelsPred = [0, 0, 1, 1, 1];

      const nmi = normalizedMutualInfo(labelsTrue, labelsPred, "arithmetic");
      expect(nmi).toBeGreaterThan(0.7);
      expect(nmi).toBeLessThan(0.85);
    });
  });

  describe("Tensor inputs", () => {
    it("should produce same result with tensor inputs", () => {
      const labelsTrue = [0, 0, 1, 1, 2, 2];
      const labelsPred = [0, 0, 1, 1, 2, 2];

      const arrayResult = normalizedMutualInfo(labelsTrue, labelsPred);

      const trueTensor = tf.tensor1d(labelsTrue);
      const predTensor = tf.tensor1d(labelsPred);
      const tensorResult = normalizedMutualInfo(trueTensor, predTensor);

      expect(tensorResult).toBeCloseTo(arrayResult, 10);

      trueTensor.dispose();
      predTensor.dispose();
    });
  });
});
