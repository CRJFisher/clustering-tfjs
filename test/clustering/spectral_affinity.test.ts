import * as tf from "../tensorflow-helper";

import { SpectralClustering } from "../../src";

describe("SpectralClustering – affinity validation", () => {
  const buildModel = () =>
    new SpectralClustering({ nClusters: 2, affinity: "precomputed" });

  it("accepts a valid precomputed symmetric affinity matrix", async () => {
    // Simple 3×3 identity-like affinity (self-similarity 1, others 0.5)
    const A = tf.tensor2d(
      [
        [1, 0.5, 0.5],
        [0.5, 1, 0.5],
        [0.5, 0.5, 1],
      ],
      [3, 3],
    );

    const model = buildModel();
    await expect(model.fit(A)).resolves.not.toThrow();
  });

  it("throws when affinity matrix is not square", async () => {
    const A = tf.tensor2d(
      [
        [1, 0.2, 0.3],
        [0.2, 1, 0.4],
      ],
      [2, 3],
    );

    const model = buildModel();
    await expect(model.fit(A)).rejects.toThrow();
  });

  it("throws when affinity matrix is asymmetric", async () => {
    const A = tf.tensor2d(
      [
        [1, 0.9],
        [0.1, 1],
      ],
      [2, 2],
    );

    const model = buildModel();
    await expect(model.fit(A)).rejects.toThrow();
  });

  it("throws when affinity matrix has negative entries", async () => {
    const A = tf.tensor2d(
      [
        [1, -0.1],
        [-0.1, 1],
      ],
      [2, 2],
    );

    const model = buildModel();
    await expect(model.fit(A)).rejects.toThrow();
  });
});

