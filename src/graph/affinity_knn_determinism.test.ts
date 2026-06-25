import * as tf from "../../test_support/tensorflow_helper";

import { compute_knn_affinity } from "./affinity";

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

    const X1 = tf.tensor2d(pts, [4, 2]);
    const X2 = tf.tensor2d(pts, [4, 2]);

    const A1 = compute_knn_affinity(X1, 2);
    const A2 = compute_knn_affinity(X2, 2);

    const equal_t = tf.equal(A1, A2);
    const all_t = equal_t.all();
    const equal = await all_t.data();
    equal_t.dispose();
    all_t.dispose();
    expect(equal[0]).toBe(1);

    A1.dispose();
    A2.dispose();
    X1.dispose();
    X2.dispose();
  });
});
