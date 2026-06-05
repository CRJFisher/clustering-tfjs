import { SpectralClustering } from "..";

describe("SpectralClustering class structure & validation", () => {
  it("instantiates with minimal valid params (defaults)", () => {
    const model = new SpectralClustering({ n_clusters: 3 });
    expect(model.params.n_clusters).toBe(3);
    expect(model.params.affinity ?? "rbf").toBe("rbf");
  });

  it("accepts nearest_neighbors with nNeighbors", () => {
    const model = new SpectralClustering({
      n_clusters: 2,
      affinity: "nearest_neighbors",
      n_neighbors: 15,
    });
    expect(model.params.affinity).toBe("nearest_neighbors");
    expect(model.params.n_neighbors).toBe(15);
  });

  it("throws for invalid nClusters", () => {
    expect(() => new SpectralClustering({ n_clusters: 0 })).toThrow();
  });

  it("throws for invalid affinity string", () => {
    expect(
      // @ts-expect-error - invalid affinity value; testing runtime validation
      () => new SpectralClustering({ n_clusters: 2, affinity: "foo" }),
    ).toThrow();
  });

  it("throws when gamma provided with nearest_neighbors", () => {
    expect(() =>
      new SpectralClustering({
        n_clusters: 2,
        affinity: "nearest_neighbors",
        gamma: 0.5,
      }),
    ).toThrow();
  });

  it("throws when nNeighbors provided with rbf affinity", () => {
    expect(() =>
      new SpectralClustering({ n_clusters: 2, n_neighbors: 5 }),
    ).toThrow();
  });
});

