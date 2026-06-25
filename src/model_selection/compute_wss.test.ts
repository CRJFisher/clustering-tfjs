import * as tf from "../../test_support/tensorflow_helper";
import { compute_wss } from "./compute_wss";

describe("compute_wss", () => {
  beforeEach(() => {
    tf.engine().startScope();
  });

  afterEach(() => {
    tf.engine().endScope();
  });

  it("returns 0 for single-point clusters", () => {
    expect(compute_wss([[0, 0], [5, 5]], [0, 1])).toBeCloseTo(0, 10);
  });

  it("returns 0 when all points in a cluster are identical", () => {
    expect(compute_wss([[3, 3], [3, 3], [3, 3]], [0, 0, 0])).toBeCloseTo(0, 10);
  });

  it("computes exact WSS for two clusters", () => {
    // Cluster 0: (0,0),(2,0) → centroid (1,0), contribution 1+1=2
    // Cluster 1: (10,0),(12,0) → centroid (11,0), contribution 1+1=2
    expect(compute_wss([[0, 0], [2, 0], [10, 0], [12, 0]], [0, 0, 1, 1])).toBeCloseTo(4, 5);
  });

  it("computes exact WSS for a single cluster", () => {
    // 1-D points [0],[4],[8]: centroid=4, WSS = 16+0+16 = 32
    expect(compute_wss([[0], [4], [8]], [0, 0, 0])).toBeCloseTo(32, 5);
  });

  it("WSS decreases as k increases for well-separated data", () => {
    const X = [
      [0, 0], [0.1, 0.1], [0.2, 0],
      [5, 5], [5.1, 5.1], [5.2, 5],
      [10, 0], [10.1, 0.1], [10.2, 0],
    ];
    const wss1 = compute_wss(X, [0, 0, 0, 0, 0, 0, 0, 0, 0]);
    const wss3 = compute_wss(X, [0, 0, 0, 1, 1, 1, 2, 2, 2]);
    expect(wss3).toBeLessThan(wss1);
  });

  it("accepts tensor data input", () => {
    const X_arr = [[0, 0], [2, 0], [10, 0], [12, 0]];
    const labels = [0, 0, 1, 1];
    const tensor = tf.tensor2d(X_arr);
    expect(compute_wss(tensor, labels)).toBeCloseTo(compute_wss(X_arr, labels), 5);
    tensor.dispose();
  });

  it("accepts tensor label input", () => {
    const X = [[0, 0], [2, 0], [10, 0], [12, 0]];
    const labels_arr = [0, 0, 1, 1];
    const labels_tensor = tf.tensor1d(labels_arr, "float32");
    expect(compute_wss(X, labels_tensor)).toBeCloseTo(compute_wss(X, labels_arr), 5);
    labels_tensor.dispose();
  });
});
