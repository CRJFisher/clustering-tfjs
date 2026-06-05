import * as tf from "../tensorflow-helper";

import {
  array_to_tensor,
  tensor_to_array,
  euclidean_distance,
  manhattan_distance,
  cosine_distance,
  pairwise_euclidean_matrix,
} from "../../src/tensor/tensor_ops";

function close_to(a: any, b: any, eps = 1e-4): boolean {
  const flat = (x: any): number[] =>
    Array.isArray(x) ? x.flat(Infinity) as number[] : [x as number];
  const arr_a = flat(a);
  const arr_b = flat(b);
  return arr_a.length === arr_b.length && arr_a.every((v, i) => Math.abs(v - arr_b[i]) < eps);
}

describe("tensor utilities", () => {
  afterEach(() => tf.engine().disposeVariables());

  it("arrayToTensor and tensorToArray round-trip", () => {
    const arr = [1, 2, 3];
    const tensor = array_to_tensor(arr);
    expect(tensor.shape).toEqual([3]);
    const back = tensor_to_array(tensor) as number[];
    expect(back).toEqual(arr);
  });

  describe("distance metrics", () => {
    const a = tf.tensor([1, 2, 3]);
    const b = tf.tensor([4, 6, 3]);

    it("euclidean distance", () => {
      const d = euclidean_distance(a, b).arraySync() as number;
      expect(close_to(d, Math.sqrt((3) ** 2 + 4 ** 2))).toBe(true);
    });

    it("manhattan distance", () => {
      const d = manhattan_distance(a, b).arraySync() as number;
      expect(d).toBe(3 + 4 + 0);
    });

    it("cosine distance", () => {
      const d = cosine_distance(a, b).arraySync() as number;
      // manual cosine similarity
      const dot = 1 * 4 + 2 * 6 + 3 * 3;
      const norm_a = Math.sqrt(1 + 4 + 9);
      const norm_b = Math.sqrt(16 + 36 + 9);
      const expected_similarity = dot / (norm_a * norm_b);
      const expected_distance = 1 - expected_similarity;
      expect(close_to(d, expected_distance)).toBe(true);
    });
  });


  describe("edge cases & broadcasting", () => {
    it("arrayToTensor respects dtype", () => {
      const t = array_to_tensor([1, 2, 3], "int32");
      expect(t.dtype).toBe("int32");
    });

    it("tensorToArray returns copy, not view", () => {
      const t = tf.tensor([1, 2, 3]);
      const arr = tensor_to_array(t) as number[];
      t.dispose(); // if arr were view, this would break; array should still be intact
      expect(arr).toEqual([1, 2, 3]);
    });

    it("broadcasting in euclideanDistance works for (n,d) vs (d)", () => {
      const big = tf.tensor2d(
        [
          [0, 0, 0],
          [1, 2, 2],
        ],
        [2, 3],
      );
      const small = tf.tensor([1, 0, 0]);
      const d = euclidean_distance(big, small).arraySync() as number[];
      const expected = [1, Math.sqrt(0 ** 2 + 2 ** 2 + 2 ** 2)];
      expect(close_to(d, expected)).toBe(true);
    });

    it("manhattanDistance is symmetric", () => {
      const p = tf.tensor([3, -4, 1]);
      const q = tf.tensor([-1, 2, 5]);
      const d1 = manhattan_distance(p, q).arraySync();
      const d2 = manhattan_distance(q, p).arraySync();
      expect(d1).toBe(d2);
    });

    it("cosine distance: identical vectors -> 0, orthogonal -> 1", () => {
      const a = tf.tensor([1, 2, 3]);
      const same = tf.tensor([1, 2, 3]);
      const ortho = tf.tensor([3, -6, 3]); // dot 0 with a? Actually dot= 1*3+2*-6+3*3=3-12+9=0

      const d_identical = cosine_distance(a, same).arraySync();
      const d_ortho = cosine_distance(a, ortho).arraySync();

      expect(close_to(d_identical, 0)).toBe(true);
      // Clamp for numerical error
      expect(close_to(d_ortho, 1)).toBe(true);
    });

    // pairwise Euclidean matrix tests moved to pairwise.test.ts
  });
});
