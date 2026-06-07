import { AgglomerativeClustering } from "..";

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
