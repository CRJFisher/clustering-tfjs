import { findOptimalClusters } from "../../src/utils/findOptimalClusters";
import { KMeans } from "../../src/clustering/kmeans";
import { makeBlobs } from "../../src/datasets/synthetic";
import * as tf from "../tensorflow-helper";

describe("findOptimalClusters", () => {
  beforeAll(async () => {
    // Set backend based on which TensorFlow version is loaded
    const backends = tf.engine().registryFactory;
    
    if ('tensorflow' in backends) {
      // Using @tensorflow/tfjs-node
      await tf.setBackend("tensorflow");
    } else {
      // Using @tensorflow/tfjs (CPU backend) - fallback for Windows CI or browser
      await tf.setBackend("cpu");
    }
  });

  afterEach(() => {
    // Clean up any tensors
    tf.engine().startScope();
    tf.engine().endScope();
  });

  it("should find optimal k for well-separated clusters", async () => {
    // Create dataset with 3 well-separated clusters
    const { X, y } = makeBlobs({
      nSamples: 150,
      nFeatures: 2,
      centers: 3,
      clusterStd: 0.5,
      randomState: 42
    });

    try {
      const result = await findOptimalClusters(X, {
        minClusters: 2,
        maxClusters: 5
      });

      // Should identify 2 or 3 as optimal (may vary by backend precision)
      expect([2, 3]).toContain(result.optimal.k);
      expect(result.optimal.silhouette).toBeGreaterThan(0.4);
      expect(result.evaluations).toHaveLength(4); // k=2,3,4,5
    } finally {
      X.dispose();
    }
  });

  it("should work with different algorithms", async () => {
    const data = [
      [1, 2], [1.5, 1.8], [5, 8], [8, 8], [1, 0.6], [9, 11],
      [1.2, 1.9], [1.4, 1.7], [5.2, 8.1], [8.1, 8.2], [0.9, 0.5], [9.1, 11.2]
    ];

    // Test with K-means
    const kmeansResult = await findOptimalClusters(data, {
      algorithm: 'kmeans',
      maxClusters: 4
    });
    expect(kmeansResult.optimal.k).toBeGreaterThanOrEqual(2);
    expect(kmeansResult.optimal.k).toBeLessThanOrEqual(4);

    // Test with Spectral
    const spectralResult = await findOptimalClusters(data, {
      algorithm: 'spectral',
      maxClusters: 4
    });
    expect(spectralResult.optimal.k).toBeGreaterThanOrEqual(2);
    expect(spectralResult.optimal.k).toBeLessThanOrEqual(4);

    // Test with Agglomerative
    const aggloResult = await findOptimalClusters(data, {
      algorithm: 'agglomerative',
      maxClusters: 4
    });
    expect(aggloResult.optimal.k).toBeGreaterThanOrEqual(2);
    expect(aggloResult.optimal.k).toBeLessThanOrEqual(4);
  });

  it("should respect min and max cluster bounds", async () => {
    const data = [[1, 2], [2, 3], [3, 4], [10, 11], [11, 12]];

    const result = await findOptimalClusters(data, {
      minClusters: 3,
      maxClusters: 4
    });

    expect(result.evaluations).toHaveLength(2); // k=3,4
    expect(result.evaluations.every(e => e.k >= 3 && e.k <= 4)).toBe(true);
  });

  it("should handle tensor input", async () => {
    const data = tf.tensor2d([[1, 2], [2, 3], [10, 11], [11, 12]]);

    const result = await findOptimalClusters(data, {
      maxClusters: 3
    });

    expect(result.optimal.k).toBe(2); // Should identify 2 clusters
    expect(result.optimal.labels).toHaveLength(4);
    
    data.dispose();
  });

  it("should use custom scoring function", async () => {
    const data = [[1, 2], [2, 3], [10, 11], [11, 12]];

    // Custom scoring that only uses silhouette
    const result = await findOptimalClusters(data, {
      maxClusters: 3,
      scoringFunction: (evaluation) => evaluation.silhouette
    });

    expect(result.optimal.combinedScore).toBe(result.optimal.silhouette);
  });

  it("should filter metrics", async () => {
    const data = [[1, 2], [2, 3], [10, 11], [11, 12]];

    // Only calculate silhouette
    const result = await findOptimalClusters(data, {
      maxClusters: 3,
      metrics: ['silhouette']
    });

    expect(result.optimal.silhouette).not.toBe(0);
    expect(result.optimal.daviesBouldin).toBe(Infinity);
    expect(result.optimal.calinskiHarabasz).toBe(0);
  });

  it("should handle edge cases", async () => {
    // Test with minimum samples - need at least 3 for 2 clusters
    const data = [[1, 2], [10, 11], [5, 6]];
    
    const result = await findOptimalClusters(data, {
      minClusters: 2,
      maxClusters: 5
    });

    // Can only have 2 clusters max with 3 samples
    expect(result.evaluations).toHaveLength(1);
    expect(result.optimal.k).toBe(2);
  });

  it("should throw on invalid parameters", async () => {
    const data = [[1, 2], [2, 3]];

    await expect(
      findOptimalClusters(data, { minClusters: 1 })
    ).rejects.toThrow('minClusters must be at least 2');

    await expect(
      findOptimalClusters(data, { minClusters: 3, maxClusters: 2 })
    ).rejects.toThrow('maxClusters must be greater than or equal to minClusters');

    // Use more samples to avoid the "not enough samples" error
    const moreData = [[1, 2], [2, 3], [10, 11], [11, 12]];
    await expect(
      findOptimalClusters(moreData, { algorithm: 'invalid' as any })
    ).rejects.toThrow('Unknown algorithm: invalid');
  });

  it("should pass algorithm parameters", async () => {
    const data = [
      [1, 2], [1.5, 1.8], [5, 8], [8, 8], [1, 0.6], [9, 11]
    ];

    const result = await findOptimalClusters(data, {
      algorithm: 'kmeans',
      algorithmParams: { nInit: 5, maxIter: 100 },
      maxClusters: 3
    });

    expect(result.optimal.k).toBeGreaterThanOrEqual(2);
    expect(result.optimal.k).toBeLessThanOrEqual(3);
  });
});