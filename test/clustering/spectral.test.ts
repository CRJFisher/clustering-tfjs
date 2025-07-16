import { SpectralClustering } from "../../src";

describe("SpectralClustering class structure & validation", () => {
  it("instantiates with minimal valid params (defaults)", () => {
    const model = new SpectralClustering({ nClusters: 3 });
    expect(model.params.nClusters).toBe(3);
    expect(model.params.affinity ?? "rbf").toBe("rbf");
  });

  it("accepts nearest_neighbors with nNeighbors", () => {
    const model = new SpectralClustering({
      nClusters: 2,
      affinity: "nearest_neighbors",
      nNeighbors: 15,
    });
    expect(model.params.affinity).toBe("nearest_neighbors");
    expect(model.params.nNeighbors).toBe(15);
  });

  it("throws for invalid nClusters", () => {
    expect(() => new SpectralClustering({ nClusters: 0 })).toThrow();
  });

  it("throws for invalid affinity string", () => {
    expect(
      () => new SpectralClustering({ nClusters: 2, affinity: "foo" as any }),
    ).toThrow();
  });

  it("throws when gamma provided with nearest_neighbors", () => {
    expect(() =>
      new SpectralClustering({
        nClusters: 2,
        affinity: "nearest_neighbors",
        gamma: 0.5,
      }),
    ).toThrow();
  });

  it("throws when nNeighbors provided with rbf affinity", () => {
    expect(() =>
      new SpectralClustering({ nClusters: 2, nNeighbors: 5 }),
    ).toThrow();
  });
});

