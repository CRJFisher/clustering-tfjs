import { describe, it, expect } from "@jest/globals";
import * as tf from "../../test_support/tensorflow_helper";
import { create_constant_eigenvector } from "./constant_eigenvector";

describe("create_constant_eigenvector", () => {
  it("returns a column vector shaped n x 1", () => {
    const affinity = tf.zeros([5, 5]) as tf.Tensor2D;
    const vec = create_constant_eigenvector(affinity);
    expect(vec.shape).toEqual([5, 1]);
    affinity.dispose();
    vec.dispose();
  });

  it("fills every entry with 1/sqrt(n)", () => {
    const n = 4;
    const affinity = tf.zeros([n, n]) as tf.Tensor2D;
    const vec = create_constant_eigenvector(affinity);
    const values = Array.from(vec.dataSync());
    const expected = 1 / Math.sqrt(n);
    for (const v of values) {
      expect(v).toBeCloseTo(expected, 6);
    }
    affinity.dispose();
    vec.dispose();
  });

  it("produces a unit-norm vector", () => {
    const affinity = tf.zeros([9, 9]) as tf.Tensor2D;
    const vec = create_constant_eigenvector(affinity);
    const norm = Math.sqrt(
      Array.from(vec.dataSync()).reduce((acc, v) => acc + v * v, 0),
    );
    expect(norm).toBeCloseTo(1.0, 6);
    affinity.dispose();
    vec.dispose();
  });

  it("depends only on the row count, not the affinity values", () => {
    // Only affinity.shape[0] is read, so the entries are irrelevant.
    const dense = tf.randomUniform([6, 6]) as tf.Tensor2D;
    const sparse = tf.zeros([6, 6]) as tf.Tensor2D;
    const from_dense = create_constant_eigenvector(dense);
    const from_sparse = create_constant_eigenvector(sparse);
    expect(Array.from(from_dense.dataSync())).toEqual(
      Array.from(from_sparse.dataSync()),
    );
    dense.dispose();
    sparse.dispose();
    from_dense.dispose();
    from_sparse.dispose();
  });

  it("handles the degenerate single-node graph", () => {
    const affinity = tf.zeros([1, 1]) as tf.Tensor2D;
    const vec = create_constant_eigenvector(affinity);
    expect(vec.shape).toEqual([1, 1]);
    expect(vec.dataSync()[0]).toBeCloseTo(1.0, 6);
    affinity.dispose();
    vec.dispose();
  });

  it("does not leak intermediate tensors", () => {
    const before = tf.memory().numTensors;
    const affinity = tf.zeros([8, 8]) as tf.Tensor2D;
    const vec = create_constant_eigenvector(affinity);
    // One input tensor plus one returned tensor remain live.
    expect(tf.memory().numTensors).toBe(before + 2);
    affinity.dispose();
    vec.dispose();
  });
});
