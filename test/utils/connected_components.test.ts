import * as tf from "../tensorflow-helper";

import {
  detectConnectedComponents,
  checkGraphConnectivity,
} from "../../src/utils/connected_components";

describe("detectConnectedComponents", () => {
  const tensors: tf.Tensor[] = [];

  function tracked<T extends tf.Tensor>(t: T): T {
    tensors.push(t);
    return t;
  }

  afterEach(() => {
    tensors.forEach((t) => t.dispose());
    tensors.length = 0;
  });

  it("finds a single component in a fully connected graph", () => {
    const affinity = tracked(
      tf.tensor2d([
        [1, 0.5, 0.3],
        [0.5, 1, 0.4],
        [0.3, 0.4, 1],
      ]),
    );
    const result = detectConnectedComponents(affinity);
    expect(result.numComponents).toBe(1);
    expect(result.isFullyConnected).toBe(true);
    expect(result.componentLabels).toEqual(new Int32Array([0, 0, 0]));
  });

  it("finds two components in a block-diagonal graph", () => {
    const affinity = tracked(
      tf.tensor2d([
        [1, 0.5, 0, 0],
        [0.5, 1, 0, 0],
        [0, 0, 1, 0.8],
        [0, 0, 0.8, 1],
      ]),
    );
    const result = detectConnectedComponents(affinity);
    expect(result.numComponents).toBe(2);
    expect(result.isFullyConnected).toBe(false);
    // Nodes 0,1 share one label; nodes 2,3 share another
    expect(result.componentLabels[0]).toBe(result.componentLabels[1]);
    expect(result.componentLabels[2]).toBe(result.componentLabels[3]);
    expect(result.componentLabels[0]).not.toBe(result.componentLabels[2]);
  });

  it("finds three disconnected components", () => {
    const affinity = tracked(
      tf.tensor2d([
        [1, 0.5, 0, 0, 0],
        [0.5, 1, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 1, 0.7],
        [0, 0, 0, 0.7, 1],
      ]),
    );
    const result = detectConnectedComponents(affinity);
    expect(result.numComponents).toBe(3);
    expect(result.isFullyConnected).toBe(false);
  });

  it("treats each isolated node as its own component", () => {
    // Identity matrix: no off-diagonal edges
    const affinity = tracked(
      tf.tensor2d([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
      ]),
    );
    const result = detectConnectedComponents(affinity);
    expect(result.numComponents).toBe(3);
    expect(result.isFullyConnected).toBe(false);
    // Each node has a unique label
    const unique = new Set(result.componentLabels);
    expect(unique.size).toBe(3);
  });

  it("handles a single-node graph", () => {
    const affinity = tracked(tf.tensor2d([[1]]));
    const result = detectConnectedComponents(affinity);
    expect(result.numComponents).toBe(1);
    expect(result.isFullyConnected).toBe(true);
    expect(result.componentLabels).toEqual(new Int32Array([0]));
  });

  it("respects custom tolerance (edges below tolerance ignored)", () => {
    // All off-diagonal values are 0.05 — below a tolerance of 0.1
    const affinity = tracked(
      tf.tensor2d([
        [1, 0.05, 0.05],
        [0.05, 1, 0.05],
        [0.05, 0.05, 1],
      ]),
    );
    const result = detectConnectedComponents(affinity, 0.1);
    expect(result.numComponents).toBe(3);
    expect(result.isFullyConnected).toBe(false);
  });

  it("uses default tolerance of 1e-2 (boundary: 0.01 is NOT > 0.01)", () => {
    // Off-diagonal value exactly at 0.01 should NOT be treated as an edge
    // because the check is strictly greater than tolerance
    const affinity = tracked(
      tf.tensor2d([
        [1, 0.01],
        [0.01, 1],
      ]),
    );
    const result = detectConnectedComponents(affinity);
    expect(result.numComponents).toBe(2);
    expect(result.isFullyConnected).toBe(false);
  });

  it("returns Int32Array for componentLabels", () => {
    const affinity = tracked(
      tf.tensor2d([
        [1, 0.5],
        [0.5, 1],
      ]),
    );
    const result = detectConnectedComponents(affinity);
    expect(result.componentLabels).toBeInstanceOf(Int32Array);
  });

  it("assigns sequential labels starting from 0", () => {
    const affinity = tracked(
      tf.tensor2d([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0.5],
        [0, 0, 0.5, 1],
      ]),
    );
    const result = detectConnectedComponents(affinity);
    const labels = Array.from(result.componentLabels);
    const unique = [...new Set(labels)].sort((a, b) => a - b);
    expect(unique[0]).toBe(0);
    for (let i = 1; i < unique.length; i++) {
      expect(unique[i]).toBe(unique[i - 1] + 1);
    }
  });
});

describe("checkGraphConnectivity", () => {
  const tensors: tf.Tensor[] = [];

  function tracked<T extends tf.Tensor>(t: T): T {
    tensors.push(t);
    return t;
  }

  afterEach(() => {
    tensors.forEach((t) => t.dispose());
    tensors.length = 0;
  });

  it("returns true for a connected graph without emitting a warning", () => {
    const warnSpy = jest.spyOn(console, "warn").mockImplementation(() => {});
    const affinity = tracked(
      tf.tensor2d([
        [1, 0.5],
        [0.5, 1],
      ]),
    );
    const result = checkGraphConnectivity(affinity);
    expect(result).toBe(true);
    expect(warnSpy).not.toHaveBeenCalled();
    warnSpy.mockRestore();
  });

  it("returns false for a disconnected graph and emits a warning", () => {
    const warnSpy = jest.spyOn(console, "warn").mockImplementation(() => {});
    const affinity = tracked(
      tf.tensor2d([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
      ]),
    );
    const result = checkGraphConnectivity(affinity);
    expect(result).toBe(false);
    expect(warnSpy).toHaveBeenCalled();
    warnSpy.mockRestore();
  });

  it("forwards custom tolerance to detection", () => {
    const warnSpy = jest.spyOn(console, "warn").mockImplementation(() => {});
    // Off-diagonal 0.05 — connected with default tolerance but not with 0.1
    const affinity = tracked(
      tf.tensor2d([
        [1, 0.05],
        [0.05, 1],
      ]),
    );
    const connectedDefault = checkGraphConnectivity(affinity);
    expect(connectedDefault).toBe(true);

    const connectedStrict = checkGraphConnectivity(affinity, 0.1);
    expect(connectedStrict).toBe(false);
    warnSpy.mockRestore();
  });
});
