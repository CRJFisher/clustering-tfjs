import { AgglomerativeClustering } from "..";
import * as tf from "../../test_support/tensorflow_helper";

describe("AgglomerativeClustering class structure & validation", () => {
  it("instantiates with minimal valid params", () => {
    const model = new AgglomerativeClustering({ n_clusters: 3 });
    expect(model.params.n_clusters).toBe(3);
    // defaults
    expect(model.params.linkage ?? "ward").toBe("ward");
  });

  it("throws for invalid n_clusters", () => {
    expect(() => new AgglomerativeClustering({ n_clusters: 0 })).toThrow();
    expect(() => new AgglomerativeClustering({ n_clusters: -2 })).toThrow();
  });

  it("throws for invalid linkage", () => {
    // @ts-expect-error - invalid linkage value; testing runtime validation
    expect(() => new AgglomerativeClustering({ n_clusters: 2, linkage: "foo" })).toThrow();
  });

  it("throws for invalid metric", () => {
    // @ts-expect-error - invalid metric value; testing runtime validation
    expect(() => new AgglomerativeClustering({ n_clusters: 2, metric: "hamming" })).toThrow();
  });

  it("throws when ward linkage used with non-euclidean metric", () => {
    expect(() =>
      new AgglomerativeClustering({ n_clusters: 2, linkage: "ward", metric: "cosine" }),
    ).toThrow();
  });

  it("fit and fit_predict run and produce expected labels", async () => {
    const X = [
      [0, 0],
      [0, 0.1],
      [5, 5],
      [5.1, 5],
    ];
    const model = new AgglomerativeClustering({ n_clusters: 2, linkage: "single" });
    const labels = await model.fit_predict(X);
    expect(labels.length).toBe(4);
    // Expect exactly two unique labels 0 and 1
    const uniq = Array.from(new Set(labels));
    expect(uniq.length).toBe(2);
  });
});

describe("AgglomerativeClustering – edge cases", () => {
  it("handles a single sample", async () => {
    const model = new AgglomerativeClustering({ n_clusters: 1, linkage: "ward" });
    const labels = await model.fit_predict([[3, 4]]);
    expect(labels).toEqual([0]);
    expect(model.children_).toEqual([]);
    expect(model.n_leaves_).toBe(1);
  });

  it("performs zero merges when n_clusters equals n_samples", async () => {
    const X = [
      [0, 0],
      [1, 1],
      [2, 2],
    ];
    const model = new AgglomerativeClustering({ n_clusters: 3, linkage: "ward" });
    const labels = await model.fit_predict(X);
    // Every sample is its own cluster — three distinct labels, no merges.
    expect(new Set(labels).size).toBe(3);
    expect(model.children_).toEqual([]);
  });

  it("ward handles coincident (zero-distance) points without NaN", async () => {
    // All-identical points exercise the Math.max(numerator, 0) clamp in the
    // Ward Lance–Williams recurrence (numerator can go slightly negative from
    // float error and the sqrt must stay real).
    const X = [
      [1, 1],
      [1, 1],
      [1, 1],
      [1, 1],
    ];
    const model = new AgglomerativeClustering({ n_clusters: 2, linkage: "ward" });
    const labels = await model.fit_predict(X);
    expect(labels.length).toBe(4);
    expect(labels.every((l) => Number.isInteger(l) && l >= 0)).toBe(true);
    expect(new Set(labels).size).toBe(2);
  });
});

// Two well-separated pairs: {0,1} close, {2,3} close, large gap between them.
const TWO_PAIRS = [
  [0, 0],
  [0, 0.1],
  [5, 5],
  [5.1, 5],
];

function euclidean_distance_matrix(X: number[][]): number[][] {
  return X.map((a) =>
    X.map((b) =>
      Math.sqrt(a.reduce((s, ai, k) => s + (ai - b[k]) ** 2, 0)),
    ),
  );
}

describe("AgglomerativeClustering – distances_", () => {
  it("records one merge distance per merge, aligned with children_", async () => {
    // n_clusters = 1 builds the full tree: n_samples - 1 = 3 merges.
    const model = new AgglomerativeClustering({ n_clusters: 1, linkage: "single" });
    await model.fit(TWO_PAIRS);

    expect(model.distances_).not.toBeNull();
    expect(model.distances_!.length).toBe(3);
    expect(model.children_!.length).toBe(model.distances_!.length);

    // Merge distances are non-decreasing for single linkage. (Distances come
    // from float32 tfjs ops, so compare with modest precision.)
    const d = model.distances_!;
    expect(d[0]).toBeCloseTo(0.1, 3);
    expect(d[1]).toBeCloseTo(0.1, 3);
    // Final merge joins the two far-apart pairs (min cross distance ~7.0).
    expect(d[2]).toBeGreaterThan(6.9);
    expect(d[1]).toBeLessThanOrEqual(d[2]);
  });

  it("distances_ has length n_samples - n_clusters", async () => {
    const model = new AgglomerativeClustering({ n_clusters: 2, linkage: "single" });
    await model.fit(TWO_PAIRS);
    expect(model.distances_!.length).toBe(2);
  });

  it("single sample yields empty distances_", async () => {
    const model = new AgglomerativeClustering({ n_clusters: 1 });
    await model.fit([[3, 4]]);
    expect(model.distances_).toEqual([]);
  });

  it("merge distances are non-decreasing for ward linkage", async () => {
    // The distance_threshold prune-prefix logic relies on monotonic merge
    // heights; verify it holds for ward (not just single linkage).
    const X = [
      [0, 0],
      [0.2, 0.1],
      [5, 5],
      [5.1, 5.2],
      [10, 0],
      [10.1, 0.2],
    ];
    const model = new AgglomerativeClustering({ n_clusters: 1, linkage: "ward" });
    await model.fit(X);
    const d = model.distances_!;
    expect(d.length).toBe(5);
    for (let i = 1; i < d.length; i++) {
      expect(d[i]).toBeGreaterThanOrEqual(d[i - 1] - 1e-9);
    }
  });
});

describe("AgglomerativeClustering – distance_threshold", () => {
  it("requires exactly one of n_clusters or distance_threshold", () => {
    expect(() => new AgglomerativeClustering({})).toThrow();
    expect(
      () => new AgglomerativeClustering({ n_clusters: 2, distance_threshold: 1 }),
    ).toThrow();
  });

  it("throws for non-positive distance_threshold", () => {
    expect(() => new AgglomerativeClustering({ distance_threshold: 0 })).toThrow();
    expect(() => new AgglomerativeClustering({ distance_threshold: -1 })).toThrow();
  });

  it("cuts the tree at the threshold (mid threshold → 2 clusters)", async () => {
    const model = new AgglomerativeClustering({
      distance_threshold: 1.0,
      linkage: "single",
    });
    const labels = await model.fit_predict(TWO_PAIRS);
    // Only the two 0.1-distance merges are below the threshold.
    expect(new Set(labels).size).toBe(2);
    expect(model.distances_!.every((d) => d < 1.0)).toBe(true);
  });

  it("merges everything when threshold exceeds all distances (1 cluster)", async () => {
    const model = new AgglomerativeClustering({
      distance_threshold: 100,
      linkage: "single",
    });
    const labels = await model.fit_predict(TWO_PAIRS);
    expect(new Set(labels).size).toBe(1);
  });

  it("merges nothing when threshold is below all distances (n clusters)", async () => {
    const model = new AgglomerativeClustering({
      distance_threshold: 0.01,
      linkage: "single",
    });
    const labels = await model.fit_predict(TWO_PAIRS);
    expect(new Set(labels).size).toBe(TWO_PAIRS.length);
    expect(model.children_).toEqual([]);
    expect(model.distances_).toEqual([]);
  });

  it("excludes a merge whose distance equals the threshold (strict <)", async () => {
    // Collinear points with exact integer single-linkage merge distances:
    // merge (0,1) at distance 1, then merge with point 2 at distance 2.
    const LINE = [
      [0, 0],
      [0, 1],
      [0, 3],
    ];

    // threshold exactly 1.0 → the distance-1 merge is at the threshold and must
    // be excluded, leaving all 3 points as singletons.
    const at_boundary = new AgglomerativeClustering({
      distance_threshold: 1.0,
      linkage: "single",
    });
    const at_labels = await at_boundary.fit_predict(LINE);
    expect(new Set(at_labels).size).toBe(3);
    expect(at_boundary.distances_).toEqual([]);

    // threshold just above 1.0 → the distance-1 merge is now included → 2 clusters.
    const above = new AgglomerativeClustering({
      distance_threshold: 1.5,
      linkage: "single",
    });
    const above_labels = await above.fit_predict(LINE);
    expect(new Set(above_labels).size).toBe(2);
    expect(above.distances_).toEqual([1]);
  });
});

describe("AgglomerativeClustering – metric 'precomputed'", () => {
  it("produces the same labels as computing distances internally", async () => {
    const linkage = "average" as const;

    const internal = new AgglomerativeClustering({ n_clusters: 2, linkage });
    const internal_labels = await internal.fit_predict(TWO_PAIRS);

    const D = euclidean_distance_matrix(TWO_PAIRS);
    const precomputed = new AgglomerativeClustering({
      n_clusters: 2,
      linkage,
      metric: "precomputed",
    });
    const precomputed_labels = await precomputed.fit_predict(D);

    expect(precomputed_labels).toEqual(internal_labels);
    // Merge distances should match the euclidean run too. The internal path
    // computes distances in float32 (tfjs) while the precomputed matrix is
    // float64, so compare with modest precision.
    expect(precomputed.distances_!.length).toBe(internal.distances_!.length);
    for (let i = 0; i < precomputed.distances_!.length; i++) {
      expect(precomputed.distances_![i]).toBeCloseTo(internal.distances_![i], 3);
    }
  });

  it("accepts a precomputed matrix passed as a tensor", async () => {
    const D = euclidean_distance_matrix(TWO_PAIRS);

    const from_array = new AgglomerativeClustering({
      n_clusters: 2,
      linkage: "average",
      metric: "precomputed",
    });
    const array_labels = await from_array.fit_predict(D);

    const D_tensor = tf.tensor2d(D);
    try {
      const from_tensor = new AgglomerativeClustering({
        n_clusters: 2,
        linkage: "average",
        metric: "precomputed",
      });
      const tensor_labels = await from_tensor.fit_predict(D_tensor);
      expect(tensor_labels).toEqual(array_labels);
    } finally {
      D_tensor.dispose();
    }
  });

  it("rejects ward linkage with precomputed metric", () => {
    expect(
      () =>
        new AgglomerativeClustering({
          n_clusters: 2,
          linkage: "ward",
          metric: "precomputed",
        }),
    ).toThrow();
  });

  it("rejects a non-square precomputed matrix", async () => {
    const model = new AgglomerativeClustering({
      n_clusters: 2,
      linkage: "average",
      metric: "precomputed",
    });
    await expect(model.fit([[0, 1, 2], [1, 0, 3]])).rejects.toThrow();
  });

  it("rejects an asymmetric precomputed matrix", async () => {
    const model = new AgglomerativeClustering({
      n_clusters: 2,
      linkage: "average",
      metric: "precomputed",
    });
    await expect(
      model.fit([
        [0, 1],
        [2, 0],
      ]),
    ).rejects.toThrow();
  });

  it("rejects a precomputed matrix with non-zero diagonal", async () => {
    const model = new AgglomerativeClustering({
      n_clusters: 2,
      linkage: "average",
      metric: "precomputed",
    });
    await expect(
      model.fit([
        [1, 1],
        [1, 1],
      ]),
    ).rejects.toThrow();
  });
});
