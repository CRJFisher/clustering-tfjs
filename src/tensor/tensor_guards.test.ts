import * as tf from "../../test_support/tensorflow_helper";
import { is_tensor } from "./tensor_guards";

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

  it("rejects an object with properties of wrong types", () => {
    const base = { dtype: "float32", shape: [3], rank: 1, dataSync: data_sync, dispose };
    expect(is_tensor({ ...base, dtype: 42 })).toBe(false);
    expect(is_tensor({ ...base, rank: "1" })).toBe(false);
    expect(is_tensor({ ...base, dataSync: "not-a-fn" })).toBe(false);
    expect(is_tensor({ ...base, dispose: 0 })).toBe(false);
  });
});
