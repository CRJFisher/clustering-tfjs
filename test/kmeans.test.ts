import { KMeans } from "../src";
import { make_random_stream } from "../src/random";
import * as tf from "./tensorflow-helper";

describe("KMeans", () => {
  const X = [
    [0, 0],
    [0.1, 0.1],
    [0.2, 0.2],
    [10, 10],
    [10.1, 10.1],
    [9.9, 9.9],
  ];

  it("should cluster two obvious blobs", async () => {
    const km = new KMeans({ n_clusters: 2, random_state: 42 });
    const labels = await km.fit_predict(X);

    const cluster0 = labels.slice(0, 3);
    const cluster1 = labels.slice(3);

    // within each true group, predicted labels must be identical
    expect(new Set(cluster0).size).toBe(1);
    expect(new Set(cluster1).size).toBe(1);

    // and across groups they must differ
    expect(cluster0[0]).not.toBe(cluster1[0]);

    expect(km.centroids_).not.toBeNull();
    expect(km.inertia_).not.toBeNull();
  });

  it("nInit=1 vs nInit=10 gives same inertia on easy data", async () => {
    const km1 = new KMeans({ n_clusters: 2, random_state: 42, n_init: 1 });
    await km1.fit(X);
    const inertia1 = km1.inertia_!;

    const km10 = new KMeans({ n_clusters: 2, random_state: 42, n_init: 10 });
    await km10.fit(X);
    const inertia10 = km10.inertia_!;

    expect(inertia10).toBeCloseTo(inertia1, 6);
  });

  it("nInit=10 should not yield higher inertia than nInit=1 on random data", async () => {
    // create random data with some overlap to make optimisation harder
    const rng = make_random_stream(42);
    const data: number[][] = Array.from({ length: 200 }, () => [rng.rand() * 10, rng.rand() * 10]);

    const km1 = new KMeans({ n_clusters: 3, random_state: 123, n_init: 1 });
    await km1.fit(data);
    const inertia1 = km1.inertia_!;

    const km10 = new KMeans({ n_clusters: 3, random_state: 123, n_init: 10 });
    await km10.fit(data);
    const inertia10 = km10.inertia_!;

    expect(inertia10).toBeLessThanOrEqual(inertia1 + 1e-6);
  });

  describe("K=1 (single cluster)", () => {
    const data = [
      [1, 2],
      [3, 4],
      [5, 6],
      [7, 8],
    ];

    it("assigns all points to cluster 0", async () => {
      const km = new KMeans({ n_clusters: 1, random_state: 42 });
      const labels = await km.fit_predict(data);

      for (const l of labels) {
        expect(l).toBe(0);
      }
      km.dispose();
    });

    it("centroid equals data mean", async () => {
      const km = new KMeans({ n_clusters: 1, random_state: 42 });
      await km.fit(data);

      const centroid_data = await km.centroids_!.array();
      const mean_x = data.reduce((s, p) => s + p[0], 0) / data.length;
      const mean_y = data.reduce((s, p) => s + p[1], 0) / data.length;

      expect(centroid_data[0][0]).toBeCloseTo(mean_x, 5);
      expect(centroid_data[0][1]).toBeCloseTo(mean_y, 5);
      km.dispose();
    });

    it("has non-negative inertia", async () => {
      const km = new KMeans({ n_clusters: 1, random_state: 42 });
      await km.fit(data);

      expect(km.inertia_).not.toBeNull();
      expect(km.inertia_!).toBeGreaterThanOrEqual(0);
      km.dispose();
    });
  });

  describe("K=nSamples", () => {
    const data = [
      [0, 0],
      [100, 0],
      [0, 100],
      [100, 100],
    ];

    it("each point gets its own cluster", async () => {
      const km = new KMeans({ n_clusters: 4, random_state: 42 });
      const labels = await km.fit_predict(data);

      expect(new Set(labels).size).toBe(4);
      km.dispose();
    });

    it("near-zero inertia", async () => {
      const km = new KMeans({ n_clusters: 4, random_state: 42 });
      await km.fit(data);

      expect(km.inertia_!).toBeLessThan(1e-6);
      km.dispose();
    });
  });

  describe("duplicate points", () => {
    it("all-identical points: valid labels, no NaN centroids", async () => {
      const data = [
        [5, 5],
        [5, 5],
        [5, 5],
        [5, 5],
      ];

      const km = new KMeans({ n_clusters: 2, random_state: 42 });
      await km.fit(data);

      expect(km.labels_).not.toBeNull();
      for (const l of km.labels_!) {
        expect(Number.isInteger(l)).toBe(true);
        expect(l).toBeGreaterThanOrEqual(0);
        expect(l).toBeLessThan(2);
      }

      const centroid_data = await km.centroids_!.array();
      for (const row of centroid_data) {
        for (const val of row) {
          expect(isNaN(val)).toBe(false);
        }
      }
      km.dispose();
    });

    it("mixed duplicates and unique: correct grouping", async () => {
      const data = [
        [0, 0], [0, 0], [0, 0],
        [10, 10], [10, 10], [10, 10],
      ];

      const km = new KMeans({ n_clusters: 2, random_state: 42 });
      const labels = await km.fit_predict(data);

      // First three should share a label
      expect(new Set(labels.slice(0, 3)).size).toBe(1);
      // Last three should share a label
      expect(new Set(labels.slice(3)).size).toBe(1);
      // The two groups should differ
      expect(labels[0]).not.toBe(labels[3]);
      km.dispose();
    });
  });

  describe("tensor input", () => {
    it("accepts tf.Tensor2D and produces correct labels", async () => {
      const data = tf.tensor2d([
        [0, 0], [0.1, 0.1],
        [10, 10], [10.1, 10.1],
      ]);

      const km = new KMeans({ n_clusters: 2, random_state: 42 });
      const labels = await km.fit_predict(data);

      expect(labels.length).toBe(4);
      expect(new Set(labels.slice(0, 2)).size).toBe(1);
      expect(new Set(labels.slice(2)).size).toBe(1);
      expect(labels[0]).not.toBe(labels[2]);

      km.dispose();
      data.dispose();
    });

    it("does not dispose caller's tensor", async () => {
      const data = tf.tensor2d([
        [0, 0], [1, 1],
        [10, 10], [11, 11],
      ]);

      const km = new KMeans({ n_clusters: 2, random_state: 42 });
      await km.fit(data);

      expect(data.isDisposed).toBe(false);

      km.dispose();
      data.dispose();
    });

    it("produces equivalent results for array and tensor input", async () => {
      const array_data = [
        [0, 0], [0.1, 0.1], [0.2, 0],
        [10, 10], [10.1, 10.1], [10.2, 10],
      ];
      const tensor_data = tf.tensor2d(array_data);

      const km1 = new KMeans({ n_clusters: 2, random_state: 42 });
      const labels1 = await km1.fit_predict(array_data);

      const km2 = new KMeans({ n_clusters: 2, random_state: 42 });
      const labels2 = await km2.fit_predict(tensor_data);

      expect(labels1).toEqual(labels2);

      km1.dispose();
      km2.dispose();
      tensor_data.dispose();
    });
  });
});
