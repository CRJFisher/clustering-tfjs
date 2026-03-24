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

describe("SpectralClusteringConsensus – additional behavior", () => {
  it("produces deterministic results with same randomState", async () => {
    const X: number[][] = [
      [0, 0], [0.1, 0.1], [0.2, 0],
      [5, 5], [5.1, 5.1], [5.2, 5],
    ];

    const model1 = new SpectralClusteringConsensus({
      nClusters: 2,
      affinity: "rbf",
      gamma: 1.0,
      randomState: 77,
      consensusRuns: 5,
    });
    await model1.fit(X);
    const labels1 = [...model1.labels_!];
    model1.dispose();

    const model2 = new SpectralClusteringConsensus({
      nClusters: 2,
      affinity: "rbf",
      gamma: 1.0,
      randomState: 77,
      consensusRuns: 5,
    });
    await model2.fit(X);
    const labels2 = [...model2.labels_!];
    model2.dispose();

    expect(labels1).toEqual(labels2);
  }, 30000);

  it("finds 3 clusters in three well-separated blobs", async () => {
    const X: number[][] = [
      // Cluster A
      [0, 0], [0.1, 0.1], [0.2, 0], [-0.1, 0.1], [0, -0.1],
      // Cluster B
      [50, 50], [50.1, 50.1], [50.2, 50], [49.9, 50.1], [50, 49.9],
      // Cluster C
      [100, 0], [100.1, 0.1], [100.2, 0], [99.9, 0.1], [100, -0.1],
    ];

    const model = new SpectralClusteringConsensus({
      nClusters: 3,
      affinity: "rbf",
      gamma: 0.01,
      randomState: 42,
      consensusRuns: 5,
    });
    await model.fit(X);

    expect(model.labels_!.length).toBe(15);
    expect(new Set(model.labels_!).size).toBe(3);

    // Points within each blob should share the same label
    const labelA = model.labels_![0];
    for (let i = 1; i < 5; i++) {
      expect(model.labels_![i]).toBe(labelA);
    }
    const labelB = model.labels_![5];
    for (let i = 6; i < 10; i++) {
      expect(model.labels_![i]).toBe(labelB);
    }
    const labelC = model.labels_![10];
    for (let i = 11; i < 15; i++) {
      expect(model.labels_![i]).toBe(labelC);
    }

    // All three labels should be distinct
    expect(new Set([labelA, labelB, labelC]).size).toBe(3);

    model.dispose();
  }, 30000);

  it("handles number[][] input (non-tensor)", async () => {
    const X: number[][] = [
      [0, 0], [0.1, 0.1],
      [5, 5], [5.1, 5.1],
    ];

    const model = new SpectralClusteringConsensus({
      nClusters: 2,
      affinity: "rbf",
      gamma: 1.0,
      randomState: 42,
      consensusRuns: 3,
    });
    await model.fit(X);

    expect(model.labels_).not.toBeNull();
    expect(model.labels_!.length).toBe(4);
    for (const l of model.labels_!) {
      expect(Number.isInteger(l)).toBe(true);
      expect(l).toBeGreaterThanOrEqual(0);
      expect(l).toBeLessThan(2);
    }

    model.dispose();
  }, 30000);

  it("can be disposed after fitting", async () => {
    const X: number[][] = [
      [0, 0], [0.1, 0.1],
      [5, 5], [5.1, 5.1],
    ];

    const model = new SpectralClusteringConsensus({
      nClusters: 2,
      affinity: "rbf",
      gamma: 1.0,
      randomState: 42,
      consensusRuns: 3,
    });
    await model.fit(X);

    expect(model.labels_).not.toBeNull();
    model.dispose();
    expect(model.labels_).toBeNull();
  }, 30000);

  it("handles minimal dataset (2 points, 2 clusters)", async () => {
    const X: number[][] = [
      [0, 0],
      [10, 10],
    ];

    const model = new SpectralClusteringConsensus({
      nClusters: 2,
      affinity: "rbf",
      gamma: 1.0,
      randomState: 42,
      consensusRuns: 3,
    });
    await model.fit(X);

    expect(model.labels_!.length).toBe(2);
    expect(new Set(model.labels_!).size).toBe(2);

    model.dispose();
  }, 30000);
});
