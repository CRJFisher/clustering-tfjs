import { SpectralClusteringConsensus } from "../../src/clustering/spectral_consensus";

describe("SpectralClusteringConsensus – normalization parity", () => {
  // Two well-separated blobs for reliable 2-cluster splitting
  const X: number[][] = [
    // Cluster 0
    [0, 0], [0.2, 0.1], [0.1, 0.2], [-0.1, 0.1], [0.15, -0.1],
    // Cluster 1
    [5, 5], [5.2, 5.1], [5.1, 5.2], [4.9, 5.1], [5.15, 4.9],
  ];

  it("produces valid labels using degree normalization", async () => {
    const model = new SpectralClusteringConsensus({
      nClusters: 2,
      affinity: "rbf",
      gamma: 1.0,
      randomState: 42,
      consensusRuns: 5,
    });

    await model.fit(X);

    expect(model.labels_).not.toBeNull();
    expect(model.labels_!.length).toBe(10);

    // All labels should be valid integers
    for (const l of model.labels_!) {
      expect(Number.isInteger(l)).toBe(true);
      expect(l).toBeGreaterThanOrEqual(0);
      expect(l).toBeLessThan(2);
    }

    // Should find exactly 2 distinct clusters
    expect(new Set(model.labels_!).size).toBe(2);

    // Points within same cluster should share labels
    const c0 = model.labels_![0];
    for (let i = 1; i < 5; i++) {
      expect(model.labels_![i]).toBe(c0);
    }
    const c1 = model.labels_![5];
    for (let i = 6; i < 10; i++) {
      expect(model.labels_![i]).toBe(c1);
    }

    // Clusters should have different labels
    expect(c0).not.toBe(c1);

    model.dispose();
  }, 30000);

  it("produces no NaN labels", async () => {
    const model = new SpectralClusteringConsensus({
      nClusters: 2,
      affinity: "rbf",
      gamma: 1.0,
      randomState: 42,
      consensusRuns: 3,
    });

    await model.fit(X);

    for (const l of model.labels_!) {
      expect(isNaN(l)).toBe(false);
      expect(isFinite(l)).toBe(true);
    }

    model.dispose();
  }, 30000);
});
