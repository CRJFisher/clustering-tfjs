import { AgglomerativeClustering } from "../../src";

describe("AgglomerativeClustering class structure & validation", () => {
  it("instantiates with minimal valid params", () => {
    const model = new AgglomerativeClustering({ nClusters: 3 });
    expect(model.params.nClusters).toBe(3);
    // defaults
    expect(model.params.linkage ?? "ward").toBe("ward");
  });

  it("throws for invalid nClusters", () => {
    expect(() => new AgglomerativeClustering({ nClusters: 0 })).toThrow();
    expect(() => new AgglomerativeClustering({ nClusters: -2 })).toThrow();
  });

  it("throws for invalid linkage", () => {
    expect(() => new AgglomerativeClustering({ nClusters: 2, linkage: "foo" as any })).toThrow();
  });

  it("throws for invalid metric", () => {
    expect(() => new AgglomerativeClustering({ nClusters: 2, metric: "hamming" as any })).toThrow();
  });

  it("throws when ward linkage used with non-euclidean metric", () => {
    expect(() =>
      new AgglomerativeClustering({ nClusters: 2, linkage: "ward", metric: "cosine" }),
    ).toThrow();
  });

  it("fit and fitPredict stubs throw informative error", async () => {
    const model = new AgglomerativeClustering({ nClusters: 2 });
    await expect(model.fit([[0, 0]])).rejects.toThrow("not implemented");
    await expect(model.fitPredict([[0, 0]])).rejects.toThrow("not implemented");
  });
});
