import * as tf from "../../test_support/tensorflow_helper";
import { adjusted_rand_index } from "./adjusted_rand_index";

describe("adjusted_rand_index", () => {
  describe("Basic functionality", () => {
    it("returns 1.0 for perfect agreement", () => {
      expect(adjusted_rand_index([0, 0, 1, 1, 2, 2], [0, 0, 1, 1, 2, 2])).toBeCloseTo(1.0, 10);
    });

    it("is permutation invariant", () => {
      expect(adjusted_rand_index([0, 0, 1, 1], [1, 1, 0, 0])).toBeCloseTo(1.0, 10);
    });

    it("is symmetric: ARI(a,b) == ARI(b,a)", () => {
      const a = [0, 0, 1, 1, 2, 2];
      const b = [0, 0, 0, 1, 1, 1];
      expect(adjusted_rand_index(a, b)).toBeCloseTo(adjusted_rand_index(b, a), 10);
    });

    it("returns negative for worse-than-random (sklearn: -0.5)", () => {
      // sklearn: adjusted_rand_score([0,0,1,1], [0,1,0,1]) = -0.5
      expect(adjusted_rand_index([0, 0, 1, 1], [0, 1, 0, 1])).toBeCloseTo(-0.5, 5);
    });

    it("returns -0.08 for balanced alternating labels (sklearn reference)", () => {
      // Two balanced clusters of 5, predictions alternate every sample.
      // Derived: index=8, expected=80/9, max=20 → ARI = -8/100 = -0.08
      const labels_true = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1];
      const labels_pred = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1];
      expect(adjusted_rand_index(labels_true, labels_pred)).toBeCloseTo(-0.08, 5);
    });
  });

  describe("sklearn reference values", () => {
    it("matches sklearn for partial agreement (≈0.5714)", () => {
      // sklearn: adjusted_rand_score([0,0,1,2], [0,0,1,1]) ≈ 0.5714
      expect(adjusted_rand_index([0, 0, 1, 2], [0, 0, 1, 1])).toBeCloseTo(0.5714, 3);
    });

    it("returns 0.0 when all true labels are identical and pred assigns each point its own cluster", () => {
      // Contingency: one row of all singletons. index=0, sum_b=0, expected=0, max=C(n,2)/2 → ARI=0
      expect(adjusted_rand_index([0, 0, 0, 0], [0, 1, 2, 3])).toBeCloseTo(0.0, 10);
    });
  });

  describe("Edge cases", () => {
    it("returns 1.0 when all samples share one cluster label in both", () => {
      expect(adjusted_rand_index([0, 0, 0], [0, 0, 0])).toBeCloseTo(1.0, 10);
    });

    it("returns 1.0 when every sample is its own singleton in both", () => {
      // Hits the denominator===0 branch: sum_a=sum_b=0, index=expected=0
      expect(adjusted_rand_index([0, 1, 2, 3], [4, 5, 6, 7])).toBeCloseTo(1.0, 10);
    });

    it("throws for empty arrays", () => {
      expect(() => adjusted_rand_index([], [])).toThrow("must not be empty");
    });

    it("throws for mismatched lengths", () => {
      expect(() => adjusted_rand_index([0, 1], [0, 1, 2])).toThrow("same length");
    });

    it("returns 1.0 for a single sample (trivially identical partitions)", () => {
      expect(adjusted_rand_index([0], [0])).toBeCloseTo(1.0, 10);
    });

    it("returns 1.0 for a single sample even with different labels", () => {
      // n_c2 = C(1,2) = 0 regardless of label values; partitions are trivially identical.
      expect(adjusted_rand_index([0], [1])).toBeCloseTo(1.0, 10);
    });
  });

  describe("Tensor inputs", () => {
    it("produces same result with tensor inputs", () => {
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
