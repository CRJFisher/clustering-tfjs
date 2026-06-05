import { AgglomerativeClustering } from "..";

describe("AgglomerativeClustering class structure & validation", () => {
  it("instantiates with minimal valid params", () => {
    const model = new AgglomerativeClustering({ n_clusters: 3 });
    expect(model.params.n_clusters).toBe(3);
    // defaults
    expect(model.params.linkage ?? "ward").toBe("ward");
  });

  it("throws for invalid nClusters", () => {
    expect(() => new AgglomerativeClustering({ n_clusters: 0 })).toThrow();
    expect(() => new AgglomerativeClustering({ n_clusters: -2 })).toThrow();
  });

  it("throws for invalid linkage", () => {
    expect(() => new AgglomerativeClustering({ n_clusters: 2, linkage: "foo" as any })).toThrow();
  });

  it("throws for invalid metric", () => {
    expect(() => new AgglomerativeClustering({ n_clusters: 2, metric: "hamming" as any })).toThrow();
  });

  it("throws when ward linkage used with non-euclidean metric", () => {
    expect(() =>
      new AgglomerativeClustering({ n_clusters: 2, linkage: "ward", metric: "cosine" }),
    ).toThrow();
  });

  it("fit and fitPredict run and produce expected labels", async () => {
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
