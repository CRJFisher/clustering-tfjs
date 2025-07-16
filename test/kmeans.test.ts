import { KMeans } from "../src";

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
    const km = new KMeans({ nClusters: 2, randomState: 42 });
    const labels = (await km.fitPredict(X)) as number[];

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
});

