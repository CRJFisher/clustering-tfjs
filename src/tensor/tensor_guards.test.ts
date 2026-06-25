import { describe, it, expect } from "@jest/globals";
import * as tf from "../../test_support/tensorflow_helper";
import { is_tensor, is_tensor_2d } from "./tensor_guards";

// Referenced (not inline) so the `dataSync` key stays an object-literal
// property rather than a method, which must literally match the tf.js API name.
const data_sync = (): Float32Array => new Float32Array(4);
const dispose = (): undefined => undefined;

describe("is_tensor", () => {
  it("accepts real 1D and 2D tensors", () => {
    const a = tf.tensor1d([1, 2, 3]);
    const b = tf.tensor2d([
      [1, 2],
      [3, 4],
    ]);
    expect(is_tensor(a)).toBe(true);
    expect(is_tensor(b)).toBe(true);
    a.dispose();
    b.dispose();
  });

  it("rejects null, undefined, and primitives", () => {
    expect(is_tensor(null)).toBe(false);
    expect(is_tensor(undefined)).toBe(false);
    expect(is_tensor(42)).toBe(false);
    expect(is_tensor("tensor")).toBe(false);
    expect(is_tensor(true)).toBe(false);
  });

  it("rejects plain objects and arrays", () => {
    expect(is_tensor({})).toBe(false);
    expect(is_tensor([1, 2, 3])).toBe(false);
  });

  it("accepts a duck-typed tensor-like object", () => {
    // The guard is structural so tensors created by a foreign tf.js build
    // (where instanceof would fail) are still recognised.
    const fake = {
      dtype: "float32",
      shape: [2, 2],
      rank: 2,
      dataSync: data_sync,
      dispose,
    };
    expect(is_tensor(fake)).toBe(true);
  });

  it("rejects an object missing any required tensor property", () => {
    const complete = {
      dtype: "float32",
      shape: [3],
      rank: 1,
      dataSync: data_sync,
      dispose,
    };
    for (const key of Object.keys(complete)) {
      const partial: Record<string, unknown> = { ...complete };
      delete partial[key];
      expect(is_tensor(partial)).toBe(false);
    }
  });

  it("rejects an object whose shape is not an array", () => {
    const bad = {
      dtype: "float32",
      shape: { 0: 2, 1: 2 },
      rank: 2,
      dataSync: data_sync,
      dispose,
    };
    expect(is_tensor(bad)).toBe(false);
  });
});

describe("is_tensor_2d", () => {
  it("accepts a rank-2 tensor", () => {
    const t = tf.tensor2d([
      [1, 2],
      [3, 4],
    ]);
    expect(is_tensor_2d(t)).toBe(true);
    t.dispose();
  });

  it("rejects tensors of other ranks", () => {
    const v = tf.tensor1d([1, 2, 3]);
    const s = tf.scalar(5);
    expect(is_tensor_2d(v)).toBe(false);
    expect(is_tensor_2d(s)).toBe(false);
    v.dispose();
    s.dispose();
  });

  it("rejects non-tensor values", () => {
    expect(is_tensor_2d(null)).toBe(false);
    expect(is_tensor_2d({ rank: 2 })).toBe(false);
  });
});
