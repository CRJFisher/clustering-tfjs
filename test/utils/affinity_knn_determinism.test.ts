import * as tf from "@tensorflow/tfjs-node";

import { compute_knn_affinity } from "../../src/utils/affinity";

describe("compute_knn_affinity – deterministic tie-breaking", () => {
  it("returns identical matrices across repeated calls when distances have ties", async () => {
    // Four points forming a square – distances along each axis are equal,
    // causing ties when k = 2.
    const pts = [
      [0, 0],
      [0, 1],
      [1, 0],
      [1, 1],
    ];

    const X = tf.tensor2d(pts, [4, 2]);

    const A1 = compute_knn_affinity(X, 2);
    const A2 = compute_knn_affinity(tf.tensor2d(pts, [4, 2]), 2);

    const equal = await tf.equal(A1, A2).all().data();
    expect(equal[0]).toBe(1);

    A1.dispose();
    A2.dispose();
  });
});
