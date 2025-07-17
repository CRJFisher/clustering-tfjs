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

  it("nInit=1 vs nInit=10 gives same inertia on easy data", async () => {
    const km1 = new KMeans({ nClusters: 2, randomState: 42, nInit: 1 });
    await km1.fit(X);
    const inertia1 = km1.inertia_!;

    const km10 = new KMeans({ nClusters: 2, randomState: 42, nInit: 10 });
    await km10.fit(X);
    const inertia10 = km10.inertia_!;

    expect(inertia10).toBeCloseTo(inertia1, 6);
  });

  it("nInit=10 should not yield higher inertia than nInit=1 on random data", async () => {
    // create random data with some overlap to make optimisation harder
    const rng = () => Math.random() * 10;
    const data: number[][] = Array.from({ length: 200 }, () => [rng(), rng()]);

    const km1 = new KMeans({ nClusters: 3, randomState: 123, nInit: 1 });
    await km1.fit(data);
    const inertia1 = km1.inertia_!;

    const km10 = new KMeans({ nClusters: 3, randomState: 123, nInit: 10 });
    await km10.fit(data);
    const inertia10 = km10.inertia_!;

    expect(inertia10).toBeLessThanOrEqual(inertia1 + 1e-6);
  });
});
