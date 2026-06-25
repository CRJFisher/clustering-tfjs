import fs from "fs";
import path from "path";
import * as tf from "../../test_support/tensorflow_helper";

import { pairwise_distance_matrix } from "./pairwise_distance";

function close_to(a: number[][], b: number[][], eps = 1e-4): boolean {
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
    const dist_mat = pairwise_distance_matrix(points, "euclidean").arraySync() as number[][];

    const expected: number[][] = [];
    const data = points.arraySync();
    for (let i = 0; i < data.length; i++) {
      expected[i] = [];
      for (let j = 0; j < data.length; j++) {
        expected[i][j] = Math.hypot(data[i][0] - data[j][0], data[i][1] - data[j][1]);
      }
    }

    expect(close_to(dist_mat, expected)).toBe(true);
  });

  it("euclidean matrix is symmetric with zero diagonal", () => {
    const pts = tf.randomUniform([5, 4]);
    const dist = pairwise_distance_matrix(pts as tf.Tensor2D, "euclidean").arraySync() as number[][];
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
    const dist = pairwise_distance_matrix(big as tf.Tensor2D, "euclidean");
    expect(dist.shape).toEqual([100, 100]);
    const duration = Date.now() - start;
    // Ensure runtime is within a reasonable bound (smoke check < 2s)
    expect(duration).toBeLessThan(2000);
  });

  it("manhattan metric produces ℓ1 distances", () => {
    const dist_mat = pairwise_distance_matrix(points, "manhattan").arraySync() as number[][];
    const data = points.arraySync();
    const expected: number[][] = [];
    for (let i = 0; i < data.length; i++) {
      expected[i] = [];
      for (let j = 0; j < data.length; j++) {
        expected[i][j] = Math.abs(data[i][0] - data[j][0]) + Math.abs(data[i][1] - data[j][1]);
      }
    }
    expect(close_to(dist_mat, expected)).toBe(true);
  });

  it("cosine metric: identical points have 0 distance, others positive", () => {
    const dist_mat = pairwise_distance_matrix(points, "cosine").arraySync() as number[][];
    for (let i = 0; i < points.shape[0]; i++) {
      expect(Math.abs(dist_mat[i][i]) < 1e-4).toBe(true);
      for (let j = 0; j < points.shape[0]; j++) {
        expect(dist_mat[i][j]).toBeGreaterThanOrEqual(0);
        expect(dist_mat[i][j]).toBeLessThanOrEqual(2); // cosine distance between 0 and 2
      }
    }
  });

  it("manhattan matrix is symmetric with zero diagonal", () => {
    const pts = tf.randomUniform([5, 4]);
    const dist = pairwise_distance_matrix(pts as tf.Tensor2D, "manhattan").arraySync() as number[][];
    const eps = 1e-3;
    for (let i = 0; i < 5; i++) {
      expect(Math.abs(dist[i][i])).toBeLessThan(eps);
      for (let j = 0; j < 5; j++) {
        expect(Math.abs(dist[i][j] - dist[j][i])).toBeLessThan(eps);
      }
    }
  });

  it("euclidean clamps coincident points to exactly 0 (no NaN from the sqrt)", () => {
    // Duplicate rows drive ‖x‖²+‖y‖²−2xᵀy slightly negative in float32; the
    // tf.maximum(.,0) guard must keep the distance real and zero.
    const dups = tf.tensor2d([
      [1, 2],
      [1, 2],
      [4, 6],
    ]);
    const dist = pairwise_distance_matrix(dups, "euclidean").arraySync() as number[][];
    expect(dist[0][1]).toBe(0);
    expect(dist[1][0]).toBe(0);
    expect(Number.isNaN(dist[0][2])).toBe(false);
    dups.dispose();
  });

  it("cosine metric encodes angle: orthogonal→1, opposite→2, scaled→0", () => {
    const vecs = tf.tensor2d([
      [1, 0],
      [0, 1],
      [-1, 0],
      [5, 0],
    ]);
    const d = pairwise_distance_matrix(vecs, "cosine").arraySync() as number[][];
    expect(d[0][1]).toBeCloseTo(1, 4); // orthogonal
    expect(d[0][2]).toBeCloseTo(2, 4); // opposite direction
    expect(d[1][2]).toBeCloseTo(1, 4); // orthogonal
    expect(d[0][3]).toBeCloseTo(0, 4); // same direction, different magnitude
    vecs.dispose();
  });

  it("cosine matrix is symmetric with zero diagonal", () => {
    const pts = tf.tensor2d([
      [1, 2, 3],
      [4, 0, 1],
      [0, 1, 0],
      [2, 2, 2],
    ]);
    const dist = pairwise_distance_matrix(pts as tf.Tensor2D, "cosine").arraySync() as number[][];
    const eps = 1e-5;
    for (let i = 0; i < 4; i++) {
      expect(Math.abs(dist[i][i])).toBeLessThan(eps);
      for (let j = 0; j < 4; j++) {
        expect(Math.abs(dist[i][j] - dist[j][i])).toBeLessThan(eps);
      }
    }
    pts.dispose();
  });

  it("throws on an unsupported metric", () => {
    // The library compiles to JS, so a runtime caller can bypass the union type.
    const fn = pairwise_distance_matrix as (
      p: tf.Tensor2D,
      m: string,
    ) => tf.Tensor2D;
    expect(() => fn(points, "hamming")).toThrow(/Unsupported metric/);
  });
});

describe("pairwise_distance_matrix – cosine parity with sklearn", () => {
  const fixture = JSON.parse(
    fs.readFileSync(
      path.join(process.cwd(), "__fixtures__", "pairwise", "cosine.json"),
      "utf-8",
    ),
  ) as { X: number[][]; cosine_distances: number[][] };

  it("matches sklearn pairwise_distances(metric='cosine')", () => {
    const X = tf.tensor2d(fixture.X);
    const D = pairwise_distance_matrix(X, "cosine");
    const arr = D.arraySync() as number[][];
    expect(close_to(arr, fixture.cosine_distances, 1e-4)).toBe(true);
    X.dispose();
    D.dispose();
  });
});
