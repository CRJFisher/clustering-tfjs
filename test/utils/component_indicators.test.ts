import * as tf from "../tensorflow-helper";

import { createComponentIndicators } from "../../src/utils/component_indicators";

describe("createComponentIndicators", () => {
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
    // 4 nodes: [0,0,1,1]
    const labels = new Int32Array([0, 0, 1, 1]);
    const result = tracked(createComponentIndicators(labels, 2, 2));
    const data = result.arraySync() as number[][];

    // Component 0 has size 2 -> value = 1/sqrt(2)
    const v = 1 / Math.sqrt(2);
    expect(data[0][0]).toBeCloseTo(v);
    expect(data[1][0]).toBeCloseTo(v);
    expect(data[0][1]).toBeCloseTo(0);
    expect(data[1][1]).toBeCloseTo(0);

    // Component 1 has size 2 -> value = 1/sqrt(2)
    expect(data[2][0]).toBeCloseTo(0);
    expect(data[3][0]).toBeCloseTo(0);
    expect(data[2][1]).toBeCloseTo(v);
    expect(data[3][1]).toBeCloseTo(v);
  });

  it("normalizes correctly for 3 components of varying sizes", () => {
    // 6 nodes: component 0 has 3, component 1 has 2, component 2 has 1
    const labels = new Int32Array([0, 0, 0, 1, 1, 2]);
    const result = tracked(createComponentIndicators(labels, 3, 3));
    const data = result.arraySync() as number[][];

    const v0 = 1 / Math.sqrt(3);
    const v1 = 1 / Math.sqrt(2);
    const v2 = 1 / Math.sqrt(1);

    // Component 0 nodes
    for (let i = 0; i < 3; i++) {
      expect(data[i][0]).toBeCloseTo(v0);
      expect(data[i][1]).toBeCloseTo(0);
      expect(data[i][2]).toBeCloseTo(0);
    }
    // Component 1 nodes
    for (let i = 3; i < 5; i++) {
      expect(data[i][0]).toBeCloseTo(0);
      expect(data[i][1]).toBeCloseTo(v1);
      expect(data[i][2]).toBeCloseTo(0);
    }
    // Component 2 node
    expect(data[5][0]).toBeCloseTo(0);
    expect(data[5][1]).toBeCloseTo(0);
    expect(data[5][2]).toBeCloseTo(v2);
  });

  it("produces columns with L2 norm equal to 1.0", () => {
    const labels = new Int32Array([0, 0, 0, 1, 1, 2]);
    const result = tracked(createComponentIndicators(labels, 3, 3));
    const data = result.arraySync() as number[][];
    const numCols = data[0].length;

    for (let c = 0; c < numCols; c++) {
      let sumSq = 0;
      for (let r = 0; r < data.length; r++) {
        sumSq += data[r][c] * data[r][c];
      }
      expect(Math.sqrt(sumSq)).toBeCloseTo(1.0);
    }
  });

  it("caps columns with maxIndicators when numComponents > maxIndicators", () => {
    // 4 components but max 2 indicators
    const labels = new Int32Array([0, 1, 2, 3]);
    const result = tracked(createComponentIndicators(labels, 4, 2));
    expect(result.shape).toEqual([4, 2]);
  });

  it("uses all components when maxIndicators >= numComponents", () => {
    const labels = new Int32Array([0, 0, 1, 1]);
    const result = tracked(createComponentIndicators(labels, 2, 10));
    expect(result.shape).toEqual([4, 2]);
  });

  it("handles a single component (all same label)", () => {
    const labels = new Int32Array([0, 0, 0, 0]);
    const result = tracked(createComponentIndicators(labels, 1, 5));
    const data = result.arraySync() as number[][];

    expect(result.shape).toEqual([4, 1]);
    const v = 1 / Math.sqrt(4);
    for (let i = 0; i < 4; i++) {
      expect(data[i][0]).toBeCloseTo(v);
    }
  });

  it("produces an identity matrix for single-node components", () => {
    // Each node is its own component
    const labels = new Int32Array([0, 1, 2]);
    const result = tracked(createComponentIndicators(labels, 3, 3));
    const data = result.arraySync() as number[][];

    // 1/sqrt(1) = 1 on diagonal, 0 elsewhere -> identity
    for (let i = 0; i < 3; i++) {
      for (let j = 0; j < 3; j++) {
        expect(data[i][j]).toBeCloseTo(i === j ? 1.0 : 0.0);
      }
    }
  });

  it("returns a float32 tensor", () => {
    const labels = new Int32Array([0, 0, 1]);
    const result = tracked(createComponentIndicators(labels, 2, 2));
    expect(result.dtype).toBe("float32");
  });

  it("returns the correct shape", () => {
    const labels = new Int32Array([0, 0, 1, 1, 2]);
    const result = tracked(createComponentIndicators(labels, 3, 3));
    expect(result.shape).toEqual([5, 3]);
  });

  it("does not leak tensors (tf.tidy)", () => {
    const labels = new Int32Array([0, 0, 1, 1]);
    const before = tf.memory().numTensors;
    const result = createComponentIndicators(labels, 2, 2);
    const after = tf.memory().numTensors;

    // Only the returned tensor should be new
    expect(after - before).toBe(1);

    result.dispose();
  });
});
