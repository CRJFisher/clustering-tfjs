import * as tf from "../tensorflow-helper";

import {
  arrayToTensor,
  tensorToArray,
  euclideanDistance,
  manhattanDistance,
  cosineDistance,
  pairwiseEuclideanMatrix,
} from "../../src/utils/tensor";

function closeTo(a: any, b: any, eps = 1e-4): boolean {
  const flat = (x: any): number[] =>
    Array.isArray(x) ? x.flat(Infinity) as number[] : [x as number];
  const arrA = flat(a);
  const arrB = flat(b);
  return arrA.length === arrB.length && arrA.every((v, i) => Math.abs(v - arrB[i]) < eps);
}

describe("tensor utilities", () => {
  afterEach(() => tf.engine().disposeVariables());

  it("arrayToTensor and tensorToArray round-trip", () => {
    const arr = [1, 2, 3];
    const tensor = arrayToTensor(arr);
    expect(tensor.shape).toEqual([3]);
    const back = tensorToArray(tensor) as number[];
    expect(back).toEqual(arr);
  });

  describe("distance metrics", () => {
    const a = tf.tensor([1, 2, 3]);
    const b = tf.tensor([4, 6, 3]);

    it("euclidean distance", () => {
      const d = euclideanDistance(a, b).arraySync() as number;
      expect(closeTo(d, Math.sqrt((3) ** 2 + 4 ** 2))).toBe(true);
    });

    it("manhattan distance", () => {
      const d = manhattanDistance(a, b).arraySync() as number;
      expect(d).toBe(3 + 4 + 0);
    });

    it("cosine distance", () => {
      const d = cosineDistance(a, b).arraySync() as number;
      // manual cosine similarity
      const dot = 1 * 4 + 2 * 6 + 3 * 3;
      const normA = Math.sqrt(1 + 4 + 9);
      const normB = Math.sqrt(16 + 36 + 9);
      const expectedSimilarity = dot / (normA * normB);
      const expectedDistance = 1 - expectedSimilarity;
      expect(closeTo(d, expectedDistance)).toBe(true);
    });
  });


  describe("edge cases & broadcasting", () => {
    it("arrayToTensor respects dtype", () => {
      const t = arrayToTensor([1, 2, 3], "int32");
      expect(t.dtype).toBe("int32");
    });

    it("tensorToArray returns copy, not view", () => {
      const t = tf.tensor([1, 2, 3]);
      const arr = tensorToArray(t) as number[];
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
      const d = euclideanDistance(big, small).arraySync() as number[];
      const expected = [1, Math.sqrt(0 ** 2 + 2 ** 2 + 2 ** 2)];
      expect(closeTo(d, expected)).toBe(true);
    });

    it("manhattanDistance is symmetric", () => {
      const p = tf.tensor([3, -4, 1]);
      const q = tf.tensor([-1, 2, 5]);
      const d1 = manhattanDistance(p, q).arraySync();
      const d2 = manhattanDistance(q, p).arraySync();
      expect(d1).toBe(d2);
    });

    it("cosine distance: identical vectors -> 0, orthogonal -> 1", () => {
      const a = tf.tensor([1, 2, 3]);
      const same = tf.tensor([1, 2, 3]);
      const ortho = tf.tensor([3, -6, 3]); // dot 0 with a? Actually dot= 1*3+2*-6+3*3=3-12+9=0

      const dIdentical = cosineDistance(a, same).arraySync();
      const dOrtho = cosineDistance(a, ortho).arraySync();

      expect(closeTo(dIdentical, 0)).toBe(true);
      // Clamp for numerical error
      expect(closeTo(dOrtho, 1)).toBe(true);
    });

    // pairwise Euclidean matrix tests moved to pairwise.test.ts
  });
});
