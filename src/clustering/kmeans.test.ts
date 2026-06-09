import fs from "fs";
import path from "path";
import { KMeans } from "..";
import { make_random_stream } from "../random";
import * as tf from "../../test_support/tensorflow_helper";

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

  it("n_init=1 vs n_init=10 gives same inertia on easy data", async () => {
    const km1 = new KMeans({ n_clusters: 2, random_state: 42, n_init: 1 });
    await km1.fit(X);
    const inertia1 = km1.inertia_!;

    const km10 = new KMeans({ n_clusters: 2, random_state: 42, n_init: 10 });
    await km10.fit(X);
    const inertia10 = km10.inertia_!;

    expect(inertia10).toBeCloseTo(inertia1, 6);
  });

  it("n_init=10 should not yield higher inertia than n_init=1 on random data", async () => {
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

describe("KMeans – centroids & predict parity with scikit-learn", () => {
  const FIXTURE_DIR = path.join(process.cwd(), "__fixtures__", "kmeans");

  function labelings_equivalent(a: number[], b: number[]): boolean {
    if (a.length !== b.length) return false;
    const fwd = new Map<number, number>();
    const rev = new Map<number, number>();
    for (let i = 0; i < a.length; i++) {
      if (fwd.has(a[i])) {
        if (fwd.get(a[i]) !== b[i]) return false;
      } else fwd.set(a[i], b[i]);
      if (rev.has(b[i])) {
        if (rev.get(b[i]) !== a[i]) return false;
      } else rev.set(b[i], a[i]);
    }
    return true;
  }

  /** Match each sklearn centroid to a distinct nearest predicted centroid. */
  function centroids_match(
    ours: number[][],
    ref: number[][],
    tol: number,
  ): boolean {
    if (ours.length !== ref.length) return false;
    const used = new Set<number>();
    for (const r of ref) {
      let best = -1;
      let best_d = Infinity;
      for (let j = 0; j < ours.length; j++) {
        if (used.has(j)) continue;
        let d = 0;
        for (let f = 0; f < r.length; f++) {
          const diff = r[f] - ours[j][f];
          d += diff * diff;
        }
        if (d < best_d) {
          best_d = d;
          best = j;
        }
      }
      if (best === -1 || Math.sqrt(best_d) > tol) return false;
      used.add(best);
    }
    return true;
  }

  const files = fs.readdirSync(FIXTURE_DIR).filter((f: string) => f.endsWith(".json"));

  for (const file of files) {
    const fixture = JSON.parse(
      fs.readFileSync(path.join(FIXTURE_DIR, file), "utf-8"),
    ) as {
      params: { n_clusters: number; random_state: number };
      X: number[][];
      labels: number[];
      cluster_centers_: number[][];
      x_test: number[][];
      predict_labels: number[];
    };

    it(`matches centroids and predict labels for ${file}`, async () => {
      const model = new KMeans({
        n_clusters: fixture.params.n_clusters,
        random_state: fixture.params.random_state,
        n_init: 10,
      });
      await model.fit(fixture.X);

      const centroids = model.get_centroids();
      expect(centroids.length).toBe(fixture.cluster_centers_.length);
      expect(centroids_match(centroids, fixture.cluster_centers_, 1e-2)).toBe(true);

      const predicted = await model.predict(fixture.x_test);
      expect(labelings_equivalent(predicted, fixture.predict_labels)).toBe(true);

      model.dispose();
    });
  }

  it("predict throws before fit", async () => {
    const model = new KMeans({ n_clusters: 2 });
    await expect(model.predict([[0, 0]])).rejects.toThrow();
  });

  it("get_centroids throws before fit", () => {
    const model = new KMeans({ n_clusters: 2 });
    expect(() => model.get_centroids()).toThrow();
  });
});

describe("KMeans – cosine (spherical) metric", () => {
  it("clusters direction-dominated data on the unit sphere", async () => {
    // Two groups pointing in opposite directions, varied magnitudes.
    const X = [
      [1, 0],
      [2, 0.1],
      [3, -0.1],
      [-1, 0],
      [-2, 0.1],
      [-3, -0.1],
    ];
    const model = new KMeans({ n_clusters: 2, metric: "cosine", random_state: 0 });
    const labels = await model.fit_predict(X);
    // First three share a label, last three share the other.
    expect(labels[0]).toBe(labels[1]);
    expect(labels[1]).toBe(labels[2]);
    expect(labels[3]).toBe(labels[4]);
    expect(labels[4]).toBe(labels[5]);
    expect(labels[0]).not.toBe(labels[3]);

    // predict reproduces fitted labels for the same data.
    const predicted = await model.predict(X);
    expect(predicted).toEqual(labels);
    model.dispose();
  });
});

describe("KMeans – JSON serialization", () => {
  const X = [
    [0, 0],
    [0.1, 0.1],
    [0.2, 0.0],
    [10, 10],
    [10.1, 9.9],
    [9.9, 10.1],
  ];

  it("to_json captures centroids, params, and inertia; from_json restores them", async () => {
    const model = new KMeans({ n_clusters: 2, random_state: 1 });
    await model.fit(X);
    const json = model.to_json();
    expect(json.centroids_).toEqual(model.get_centroids());
    expect(json.params.n_clusters).toBe(2);
    expect(json.inertia_).toBe(model.inertia_);

    // Survives a real JSON round-trip.
    const restored = KMeans.from_json(JSON.parse(JSON.stringify(json)));
    expect(restored.get_centroids()).toEqual(model.get_centroids());
    expect(restored.inertia_).toBe(model.inertia_);
    expect(restored.params.n_clusters).toBe(2);
  });

  it("predict after from_json reproduces the original assignment exactly", async () => {
    const model = new KMeans({ n_clusters: 2, random_state: 1 });
    await model.fit(X);
    const original = await model.predict(X);

    const restored = KMeans.from_json(JSON.parse(JSON.stringify(model.to_json())));
    const after = await restored.predict(X);
    expect(after).toEqual(original);
  });

  it("to_json throws before fit", () => {
    const model = new KMeans({ n_clusters: 2 });
    expect(() => model.to_json()).toThrow();
  });
});
