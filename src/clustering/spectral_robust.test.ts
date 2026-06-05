import * as tf from "../../test_support/tensorflow_helper";

import { SpectralClustering, DataMatrix } from "..";

/* -------------------------------------------------------------------------- */
/*                      Helper to build a simple toy dataset                  */
/* -------------------------------------------------------------------------- */

function make_two_blobs(): tf.Tensor2D {
  // Four points around (0,0) and four around (5,5)
  const pts = [
    [-0.2, 0.1],
    [0.1, -0.1],
    [0.2, 0.2],
    [-0.1, -0.2],
    [5.0, 5.1],
    [5.2, 4.9],
    [4.8, 5.0],
    [5.1, 5.2],
  ];
  return tf.tensor2d(pts, [pts.length, 2]);
}

describe("SpectralClustering – robustness", () => {
  it("correctly separates two obvious blobs", async () => {
    const X = make_two_blobs();

    const model = new SpectralClustering({ n_clusters: 2, random_state: 42 });
    const labels = await model.fit_predict(X);

    // Expect exactly 2 unique labels
    const unique = Array.from(new Set(labels));
    expect(unique.length).toBe(2);

    // First four samples should share the same label, last four another
    const first_label = labels[0];
    for (let i = 1; i < 4; i++) {
      expect(labels[i]).toBe(first_label);
    }

    const second_label = labels[4];
    for (let i = 5; i < 8; i++) {
      expect(labels[i]).toBe(second_label);
    }

    expect(first_label).not.toBe(second_label);

    model.dispose();
  });

  it("throws on zero affinity matrix", async () => {
    const Z = tf.tensor2d(Array(9).fill(0), [3, 3]);
    const model = new SpectralClustering({ affinity: "precomputed", n_clusters: 2 });
    await expect(model.fit(Z)).rejects.toThrow();
  });

  it("throws when callable affinity returns non-square matrix", async () => {
    const X = make_two_blobs();
    const bad_callable = (_: DataMatrix) =>
      tf.tensor2d(Array(X.shape[0] * (X.shape[0] + 1)).fill(0), [
        X.shape[0],
        X.shape[0] + 1,
      ]);

    const model = new SpectralClustering({ affinity: bad_callable, n_clusters: 2 });
    await expect(model.fit(X)).rejects.toThrow();
  });

  it("throws when precomputed affinity is non-square", async () => {
    const A = tf.tensor2d(Array(6).fill(0), [2, 3]);
    const model = new SpectralClustering({ n_clusters: 2, affinity: "precomputed" });
    await expect(model.fit(A)).rejects.toThrow();
  });
});
