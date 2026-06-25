import * as tf from "../../test_support/tensorflow_helper";
import { normalized_mutual_info } from "./normalized_mutual_info";

describe("normalized_mutual_info", () => {
  describe("Basic functionality", () => {
    it("returns 1.0 for perfect agreement", () => {
      expect(normalized_mutual_info([0, 0, 1, 1, 2, 2], [0, 0, 1, 1, 2, 2])).toBeCloseTo(1.0, 10);
    });

    it("is permutation invariant", () => {
      expect(normalized_mutual_info([0, 0, 1, 1], [1, 1, 0, 0])).toBeCloseTo(1.0, 10);
    });

    it("is symmetric: NMI(a,b) == NMI(b,a)", () => {
      const a = [0, 0, 1, 1, 2, 2];
      const b = [0, 0, 0, 1, 1, 1];
      expect(normalized_mutual_info(a, b)).toBeCloseTo(normalized_mutual_info(b, a), 10);
    });

    it("returns 0 when true labels are all identical (independent clustering)", () => {
      // H(true)=0, MI=0, normalizer=(H(true)+H(pred))/2 > 0 → NMI = 0
      expect(normalized_mutual_info([0, 0, 0, 0], [0, 1, 2, 3])).toBeCloseTo(0, 5);
    });

    it("result is always in [0, 1]", () => {
      const nmi = normalized_mutual_info([0, 0, 1, 1, 2, 2], [0, 1, 1, 2, 2, 0]);
      expect(nmi).toBeGreaterThanOrEqual(0);
      expect(nmi).toBeLessThanOrEqual(1);
    });
  });

  describe("Average methods", () => {
    it("defaults to arithmetic average", () => {
      const labels_true = [0, 0, 1, 2];
      const labels_pred = [0, 0, 1, 1];
      expect(normalized_mutual_info(labels_true, labels_pred)).toBeCloseTo(
        normalized_mutual_info(labels_true, labels_pred, "arithmetic"), 10,
      );
    });

    it("each averaging method yields the correct exact sklearn value", () => {
      // labels_true=[0,0,1,1,2], labels_pred=[0,0,1,1,1]
      // pred is a coarsening of true (true clusters 1 and 2 merge into pred cluster 1)
      // MI = H(pred) = 0.6730, H(true) = 1.0549
      // arithmetic ≈ 0.7791, geometric ≈ 0.7988, min = 1.0, max ≈ 0.6381
      const labels_true = [0, 0, 1, 1, 2];
      const labels_pred = [0, 0, 1, 1, 1];

      expect(normalized_mutual_info(labels_true, labels_pred, "arithmetic")).toBeCloseTo(0.7791, 3);
      expect(normalized_mutual_info(labels_true, labels_pred, "geometric")).toBeCloseTo(0.7988, 3);
      expect(normalized_mutual_info(labels_true, labels_pred, "min")).toBeCloseTo(1.0, 5);
      expect(normalized_mutual_info(labels_true, labels_pred, "max")).toBeCloseTo(0.6381, 3);
    });

    it("normalizer ordering: NMI_min >= NMI_geometric >= NMI_arithmetic >= NMI_max", () => {
      // Entropy inequality: H_min <= sqrt(H*H) <= (H+H)/2 <= H_max
      // Dividing MI by a larger normalizer yields a smaller NMI.
      const a = [0, 0, 1, 1, 2];
      const b = [0, 0, 1, 1, 1];
      const nmi_min = normalized_mutual_info(a, b, "min");
      const nmi_geo = normalized_mutual_info(a, b, "geometric");
      const nmi_ari = normalized_mutual_info(a, b, "arithmetic");
      const nmi_max = normalized_mutual_info(a, b, "max");
      expect(nmi_min).toBeGreaterThanOrEqual(nmi_geo - 1e-10);
      expect(nmi_geo).toBeGreaterThanOrEqual(nmi_ari - 1e-10);
      expect(nmi_ari).toBeGreaterThanOrEqual(nmi_max - 1e-10);
    });
  });

  describe("Edge cases", () => {
    it("returns 1.0 when both are single cluster (normalizer=0, MI=0)", () => {
      expect(normalized_mutual_info([0, 0, 0], [0, 0, 0])).toBeCloseTo(1.0, 10);
    });

    it("returns 1.0 for a single sample", () => {
      expect(normalized_mutual_info([0], [0])).toBeCloseTo(1.0, 10);
    });

    it("throws for empty arrays", () => {
      expect(() => normalized_mutual_info([], [])).toThrow("must not be empty");
    });

    it("throws for mismatched lengths", () => {
      expect(() => normalized_mutual_info([0, 1], [0, 1, 2])).toThrow("same length");
    });
  });

  describe("Tensor inputs", () => {
    it("produces same result with tensor inputs", () => {
      const labels_true = [0, 0, 1, 1, 2, 2];
      const labels_pred = [0, 0, 1, 1, 2, 2];
      const array_result = normalized_mutual_info(labels_true, labels_pred);
      const true_tensor = tf.tensor1d(labels_true);
      const pred_tensor = tf.tensor1d(labels_pred);
      expect(normalized_mutual_info(true_tensor, pred_tensor)).toBeCloseTo(array_result, 10);
      true_tensor.dispose();
      pred_tensor.dispose();
    });
  });
});
