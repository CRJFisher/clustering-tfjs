import { KMeans } from "../../src/clustering/kmeans";

describe("KMeans empty cluster handling", () => {
  it("should reassign empty clusters to farthest points like sklearn", async () => {
    // Create a dataset that will likely produce empty clusters
    // 3 tight groups, but asking for 5 clusters
    const X = [
      // Group 1: 20 points near origin
      ...Array.from({ length: 20 }, () => [
        Math.random() * 0.5,
        Math.random() * 0.5,
      ]),
      // Group 2: 5 points at (5, 5)
      ...Array.from({ length: 5 }, () => [
        5 + Math.random() * 0.5,
        5 + Math.random() * 0.5,
      ]),
      // Group 3: 5 points at (5, -5)
      ...Array.from({ length: 5 }, () => [
        5 + Math.random() * 0.5,
        -5 + Math.random() * 0.5,
      ]),
    ];

    const km = new KMeans({
      nClusters: 5,
      nInit: 1,
      randomState: 42,
    });

    await km.fit(X);

    // Check that all clusters have been assigned
    const labels = km.labels_ as number[];
    const uniqueLabels = new Set(labels);
    expect(uniqueLabels.size).toBe(5);

    // Check that labels are valid
    expect(labels).toHaveLength(30);
    labels.forEach((label) => {
      expect(label).toBeGreaterThanOrEqual(0);
      expect(label).toBeLessThan(5);
    });

    // Verify centroids are reasonable (no NaN or infinity)
    const centroids = await km.centroids_!.array();
    centroids.forEach((centroid) => {
      centroid.forEach((value) => {
        expect(value).not.toBeNaN();
        expect(Math.abs(value)).toBeLessThan(Infinity);
      });
    });
  });

});