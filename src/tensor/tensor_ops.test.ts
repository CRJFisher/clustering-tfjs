import * as tf from "../../test_support/tensorflow_helper";

import {
  euclidean_distance,
  manhattan_distance,
  cosine_distance,
} from "./tensor_ops";

function close_to(a: unknown, b: unknown, eps = 1e-4): boolean {
  const flat = (x: unknown): number[] =>
    Array.isArray(x) ? (x.flat(Infinity) as number[]) : [x as number];
  const arr_a = flat(a);
  const arr_b = flat(b);
  return (
    arr_a.length === arr_b.length &&
    arr_a.every((v, i) => Math.abs(v - arr_b[i]) < eps)
  );
}

describe("euclidean_distance", () => {
  it("computes sqrt(sum of squared diffs) along the last axis", () => {
    const a = tf.tensor([1, 2, 3]);
    const b = tf.tensor([4, 6, 3]);
    const d = euclidean_distance(a, b).arraySync() as number;
    expect(close_to(d, Math.sqrt(3 ** 2 + 4 ** 2))).toBe(true);
    a.dispose();
    b.dispose();
  });

  it("broadcasts (n,d) against (d)", () => {
    const big = tf.tensor2d([[0, 0, 0], [1, 2, 2]], [2, 3]);
    const small = tf.tensor([1, 0, 0]);
    const d = euclidean_distance(big, small).arraySync() as number[];
    expect(close_to(d, [1, Math.sqrt(0 ** 2 + 2 ** 2 + 2 ** 2)])).toBe(true);
    big.dispose();
    small.dispose();
  });
});

describe("manhattan_distance", () => {
  it("sums absolute diffs along the last axis", () => {
    const a = tf.tensor([1, 2, 3]);
    const b = tf.tensor([4, 6, 3]);
    const d = manhattan_distance(a, b).arraySync() as number;
    expect(d).toBe(3 + 4 + 0);
    a.dispose();
    b.dispose();
  });

  it("is symmetric", () => {
    const p = tf.tensor([3, -4, 1]);
    const q = tf.tensor([-1, 2, 5]);
    const d1 = manhattan_distance(p, q).arraySync();
    const d2 = manhattan_distance(q, p).arraySync();
    expect(d1).toBe(d2);
    p.dispose();
    q.dispose();
  });

  it("broadcasts (n,d) against (d)", () => {
    const big = tf.tensor2d([[1, 0], [0, 1]], [2, 2]);
    const small = tf.tensor([1, 0]);
    const d = manhattan_distance(big, small).arraySync() as number[];
    expect(d).toEqual([0, 2]);
    big.dispose();
    small.dispose();
  });
});

describe("cosine_distance", () => {
  it("computes 1 - cosine_similarity", () => {
    const a = tf.tensor([1, 2, 3]);
    const b = tf.tensor([4, 6, 3]);
    const d = cosine_distance(a, b).arraySync() as number;
    const dot = 1 * 4 + 2 * 6 + 3 * 3;
    const norm_a = Math.sqrt(1 + 4 + 9);
    const norm_b = Math.sqrt(16 + 36 + 9);
    expect(close_to(d, 1 - dot / (norm_a * norm_b))).toBe(true);
    a.dispose();
    b.dispose();
  });

  it("identical vectors -> 0, orthogonal vectors -> 1", () => {
    const a = tf.tensor([1, 2, 3]);
    const same = tf.tensor([1, 2, 3]);
    const ortho = tf.tensor([3, -6, 3]); // dot with a: 3 + -12 + 9 = 0
    expect(close_to(cosine_distance(a, same).arraySync(), 0)).toBe(true);
    expect(close_to(cosine_distance(a, ortho).arraySync(), 1)).toBe(true);
    a.dispose();
    same.dispose();
    ortho.dispose();
  });

  it("opposite vectors -> 2", () => {
    const a = tf.tensor([1, 2, 3]);
    const opp = tf.tensor([-1, -2, -3]);
    expect(close_to(cosine_distance(a, opp).arraySync(), 2)).toBe(true);
    a.dispose();
    opp.dispose();
  });

  it("stays finite for a zero vector (eps guard)", () => {
    const zero = tf.tensor([0, 0, 0]);
    const v = tf.tensor([1, 1, 1]);
    const d = cosine_distance(zero, v).arraySync() as number;
    expect(Number.isFinite(d)).toBe(true);
    expect(close_to(d, 1)).toBe(true);
    zero.dispose();
    v.dispose();
  });

  it("broadcasts (n,d) against (d)", () => {
    const big = tf.tensor2d([[1, 0], [0, 1]], [2, 2]);
    const small = tf.tensor([1, 0]);
    const d = cosine_distance(big, small).arraySync() as number[];
    expect(close_to(d, [0, 1])).toBe(true);
    big.dispose();
    small.dispose();
  });
});
