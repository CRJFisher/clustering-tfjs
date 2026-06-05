import { describe, it, expect, beforeEach, afterEach } from "@jest/globals";
import * as tf from "../tensorflow-helper";
import { compute_wss } from "../../src/model_selection/compute_wss";

describe("computeWss", () => {
  beforeEach(() => {
    tf.engine().startScope();
  });

  afterEach(() => {
    tf.engine().endScope();
  });

  it("should compute 0 WSS for single-point clusters", () => {
    const X = [[0, 0], [5, 5]];
    const labels = [0, 1];

    // Each cluster has one point, so WSS = 0
    expect(compute_wss(X, labels)).toBeCloseTo(0, 10);
  });

  it("should compute correct WSS for known data", () => {
    // Two clusters:
    // Cluster 0: (0,0), (2,0) -> centroid (1,0), WSS = 1+1 = 2
    // Cluster 1: (10,0), (12,0) -> centroid (11,0), WSS = 1+1 = 2
    // Total WSS = 4
    const X = [[0, 0], [2, 0], [10, 0], [12, 0]];
    const labels = [0, 0, 1, 1];

    expect(compute_wss(X, labels)).toBeCloseTo(4, 5);
  });

  it("should decrease as k increases for well-separated data", () => {
    // With 3 natural clusters, k=3 should have lower WSS than k=1 (all in one)
    const X = [
      [0, 0], [0.1, 0.1], [0.2, 0],
      [5, 5], [5.1, 5.1], [5.2, 5],
      [10, 0], [10.1, 0.1], [10.2, 0],
    ];

    // All in one cluster
    const wss1 = compute_wss(X, [0, 0, 0, 0, 0, 0, 0, 0, 0]);
    // Three clusters
    const wss3 = compute_wss(X, [0, 0, 0, 1, 1, 1, 2, 2, 2]);

    expect(wss3).toBeLessThan(wss1);
  });

  it("should work with tensor input", () => {
    const X = [[0, 0], [2, 0], [10, 0], [12, 0]];
    const labels = [0, 0, 1, 1];

    const array_wss = compute_wss(X, labels);

    const tensor = tf.tensor2d(X);
    const tensor_wss = compute_wss(tensor, labels);

    expect(tensor_wss).toBeCloseTo(array_wss, 5);

    tensor.dispose();
  });
});
