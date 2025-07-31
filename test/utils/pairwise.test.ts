import * as tf from "../tensorflow-helper";

import { pairwiseDistanceMatrix } from "../../src/utils/pairwise_distance";

function closeTo(a: number[][], b: number[][], eps = 1e-4): boolean {
  for (let i = 0; i < a.length; i++) {
    for (let j = 0; j < a[0].length; j++) {
      if (Math.abs(a[i][j] - b[i][j]) > eps) return false;
    }
  }
  return true;
}

describe("pairwiseDistanceMatrix", () => {
  const points = tf.tensor2d(
    [
      [0, 0],
      [3, 4],
      [6, 8],
    ],
    [3, 2],
  );

  afterEach(() => tf.engine().disposeVariables());

  it("euclidean metric matches pairwiseEuclideanMatrix", () => {
    const distMat = pairwiseDistanceMatrix(points, "euclidean").arraySync() as number[][];

    const expected: number[][] = [];
    const data = points.arraySync();
    for (let i = 0; i < data.length; i++) {
      expected[i] = [];
      for (let j = 0; j < data.length; j++) {
        expected[i][j] = Math.hypot(data[i][0] - data[j][0], data[i][1] - data[j][1]);
      }
    }

    expect(closeTo(distMat, expected)).toBe(true);
  });

  it("euclidean matrix is symmetric with zero diagonal", () => {
    const pts = tf.randomUniform([5, 4]);
    const dist = pairwiseDistanceMatrix(pts as tf.Tensor2D, "euclidean").arraySync() as number[][];
    const eps = 1e-3;
    for (let i = 0; i < 5; i++) {
      expect(Math.abs(dist[i][i]) < eps).toBe(true);
      for (let j = 0; j < 5; j++) {
        expect(Math.abs(dist[i][j] - dist[j][i]) < eps).toBe(true);
      }
    }
  });

  it("computes matrix for 100x50 without crashing (performance smoke)", () => {
    const big = tf.randomUniform([100, 50]);
    const start = Date.now();
    const dist = pairwiseDistanceMatrix(big as tf.Tensor2D, "euclidean");
    expect(dist.shape).toEqual([100, 100]);
    const duration = Date.now() - start;
    // Ensure runtime is within a reasonable bound (smoke check < 2s)
    expect(duration).toBeLessThan(2000);
  });

  it("manhattan metric produces ℓ1 distances", () => {
    const distMat = pairwiseDistanceMatrix(points, "manhattan").arraySync() as number[][];
    const data = points.arraySync();
    const expected: number[][] = [];
    for (let i = 0; i < data.length; i++) {
      expected[i] = [];
      for (let j = 0; j < data.length; j++) {
        expected[i][j] = Math.abs(data[i][0] - data[j][0]) + Math.abs(data[i][1] - data[j][1]);
      }
    }
    expect(closeTo(distMat, expected)).toBe(true);
  });

  it("cosine metric: identical points have 0 distance, others positive", () => {
    const distMat = pairwiseDistanceMatrix(points, "cosine").arraySync() as number[][];
    for (let i = 0; i < points.shape[0]; i++) {
      expect(Math.abs(distMat[i][i]) < 1e-4).toBe(true);
      for (let j = 0; j < points.shape[0]; j++) {
        expect(distMat[i][j]).toBeGreaterThanOrEqual(0);
        expect(distMat[i][j]).toBeLessThanOrEqual(2); // cosine distance between 0 and 2
      }
    }
  });
});
