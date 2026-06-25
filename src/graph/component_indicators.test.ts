import * as tf from "../../test_support/tensorflow_helper";

import { create_component_indicators } from "./component_indicators";

describe("create_component_indicators", () => {
  const tensors: tf.Tensor[] = [];

  function tracked<T extends tf.Tensor>(t: T): T {
    tensors.push(t);
    return t;
  }

  afterEach(() => {
    tensors.forEach((t) => t.dispose());
    tensors.length = 0;
  });

  it("produces correct values for 2 equal-size components", () => {
    const labels = new Int32Array([0, 0, 1, 1]);
    const result = tracked(create_component_indicators(labels, 2, 2));
    const data = result.arraySync() as number[][];

    const v = 1 / Math.sqrt(2);
    expect(data[0][0]).toBeCloseTo(v);
    expect(data[1][0]).toBeCloseTo(v);
    expect(data[0][1]).toBeCloseTo(0);
    expect(data[1][1]).toBeCloseTo(0);

    expect(data[2][0]).toBeCloseTo(0);
    expect(data[3][0]).toBeCloseTo(0);
    expect(data[2][1]).toBeCloseTo(v);
    expect(data[3][1]).toBeCloseTo(v);
  });

  it("normalizes correctly for 3 components of varying sizes", () => {
    const labels = new Int32Array([0, 0, 0, 1, 1, 2]);
    const result = tracked(create_component_indicators(labels, 3, 3));
    const data = result.arraySync() as number[][];

    const v0 = 1 / Math.sqrt(3);
    const v1 = 1 / Math.sqrt(2);
    const v2 = 1 / Math.sqrt(1);

    for (let i = 0; i < 3; i++) {
      expect(data[i][0]).toBeCloseTo(v0);
      expect(data[i][1]).toBeCloseTo(0);
      expect(data[i][2]).toBeCloseTo(0);
    }
    for (let i = 3; i < 5; i++) {
      expect(data[i][0]).toBeCloseTo(0);
      expect(data[i][1]).toBeCloseTo(v1);
      expect(data[i][2]).toBeCloseTo(0);
    }
    expect(data[5][0]).toBeCloseTo(0);
    expect(data[5][1]).toBeCloseTo(0);
    expect(data[5][2]).toBeCloseTo(v2);
  });

  it("produces columns with L2 norm equal to 1.0", () => {
    const labels = new Int32Array([0, 0, 0, 1, 1, 2]);
    const result = tracked(create_component_indicators(labels, 3, 3));
    const data = result.arraySync() as number[][];
    const num_cols = data[0].length;

    for (let c = 0; c < num_cols; c++) {
      let sum_sq = 0;
      for (let r = 0; r < data.length; r++) {
        sum_sq += data[r][c] * data[r][c];
      }
      expect(Math.sqrt(sum_sq)).toBeCloseTo(1.0);
    }
  });

  it("caps columns at max_indicators; nodes in excess components get zero rows", () => {
    const labels = new Int32Array([0, 1, 2, 3]);
    const result = tracked(create_component_indicators(labels, 4, 2));
    expect(result.shape).toEqual([4, 2]);

    const data = result.arraySync() as number[][];
    expect(data[0]).toEqual([1, 0]);
    expect(data[1]).toEqual([0, 1]);
    expect(data[2]).toEqual([0, 0]);
    expect(data[3]).toEqual([0, 0]);
  });

  it("uses all components when max_indicators >= num_components", () => {
    const labels = new Int32Array([0, 0, 1, 1]);
    const result = tracked(create_component_indicators(labels, 2, 10));
    expect(result.shape).toEqual([4, 2]);
  });

  it("handles a single component (all same label)", () => {
    const labels = new Int32Array([0, 0, 0, 0]);
    const result = tracked(create_component_indicators(labels, 1, 5));
    const data = result.arraySync() as number[][];

    expect(result.shape).toEqual([4, 1]);
    const v = 1 / Math.sqrt(4);
    for (let i = 0; i < 4; i++) {
      expect(data[i][0]).toBeCloseTo(v);
    }
  });

  it("produces an identity matrix for single-node components", () => {
    const labels = new Int32Array([0, 1, 2]);
    const result = tracked(create_component_indicators(labels, 3, 3));
    const data = result.arraySync() as number[][];

    for (let i = 0; i < 3; i++) {
      for (let j = 0; j < 3; j++) {
        expect(data[i][j]).toBeCloseTo(i === j ? 1.0 : 0.0);
      }
    }
  });

  it("returns a float32 tensor", () => {
    const labels = new Int32Array([0, 0, 1]);
    const result = tracked(create_component_indicators(labels, 2, 2));
    expect(result.dtype).toBe("float32");
  });

  it("returns the correct shape", () => {
    const labels = new Int32Array([0, 0, 1, 1, 2]);
    const result = tracked(create_component_indicators(labels, 3, 3));
    expect(result.shape).toEqual([5, 3]);
  });

  it("does not leak tensors (tf.tidy)", () => {
    const labels = new Int32Array([0, 0, 1, 1]);
    const before = tf.memory().numTensors;
    const result = create_component_indicators(labels, 2, 2);
    const after = tf.memory().numTensors;

    expect(after - before).toBe(1);

    result.dispose();
  });
});
