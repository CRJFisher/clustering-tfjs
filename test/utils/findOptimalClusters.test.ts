import { find_optimal_clusters } from "../../src/model_selection/find_optimal_clusters";
import { KMeans } from "../../src/clustering/kmeans";
import { make_blobs } from "../../src/datasets/synthetic";
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

  beforeEach(() => {
    tf.engine().startScope();
  });

  afterEach(() => {
    tf.engine().endScope();
  });

  it("should find optimal k for well-separated clusters", async () => {
    // Create dataset with 3 well-separated clusters
    const { X, y } = make_blobs({
      n_samples: 150,
      n_features: 2,
      centers: 3,
      cluster_std: 0.5,
      random_state: 42
    });

    try {
      const result = await find_optimal_clusters(X, {
        min_clusters: 2,
        max_clusters: 5
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
    const kmeans_result = await find_optimal_clusters(data, {
      algorithm: 'kmeans',
      max_clusters: 4
    });
    expect(kmeans_result.optimal.k).toBeGreaterThanOrEqual(2);
    expect(kmeans_result.optimal.k).toBeLessThanOrEqual(4);

    // Test with Spectral
    const spectral_result = await find_optimal_clusters(data, {
      algorithm: 'spectral',
      max_clusters: 4
    });
    expect(spectral_result.optimal.k).toBeGreaterThanOrEqual(2);
    expect(spectral_result.optimal.k).toBeLessThanOrEqual(4);

    // Test with Agglomerative
    const agglo_result = await find_optimal_clusters(data, {
      algorithm: 'agglomerative',
      max_clusters: 4
    });
    expect(agglo_result.optimal.k).toBeGreaterThanOrEqual(2);
    expect(agglo_result.optimal.k).toBeLessThanOrEqual(4);
  });

  it("should respect min and max cluster bounds", async () => {
    const data = [[1, 2], [2, 3], [3, 4], [10, 11], [11, 12]];

    const result = await find_optimal_clusters(data, {
      min_clusters: 3,
      max_clusters: 4
    });

    expect(result.evaluations).toHaveLength(2); // k=3,4
    expect(result.evaluations.every(e => e.k >= 3 && e.k <= 4)).toBe(true);
  });

  it("should handle tensor input", async () => {
    const data = tf.tensor2d([[1, 2], [2, 3], [10, 11], [11, 12]]);

    const result = await find_optimal_clusters(data, {
      max_clusters: 3
    });

    expect(result.optimal.k).toBe(2); // Should identify 2 clusters
    expect(result.optimal.labels).toHaveLength(4);
    
    data.dispose();
  });

  it("should use custom scoring function", async () => {
    const data = [[1, 2], [2, 3], [10, 11], [11, 12]];

    // Custom scoring that only uses silhouette
    const result = await find_optimal_clusters(data, {
      max_clusters: 3,
      scoring_function: (evaluation) => evaluation.silhouette
    });

    expect(result.optimal.combined_score).toBe(result.optimal.silhouette);
  });

  it("should filter metrics", async () => {
    const data = [[1, 2], [2, 3], [10, 11], [11, 12]];

    // Only calculate silhouette
    const result = await find_optimal_clusters(data, {
      max_clusters: 3,
      metrics: ['silhouette']
    });

    expect(result.optimal.silhouette).not.toBe(0);
    expect(result.optimal.davies_bouldin).toBe(Infinity);
    expect(result.optimal.calinski_harabasz).toBe(0);
  });

  it("should handle edge cases", async () => {
    // Test with minimum samples - need at least 3 for 2 clusters
    const data = [[1, 2], [10, 11], [5, 6]];
    
    const result = await find_optimal_clusters(data, {
      min_clusters: 2,
      max_clusters: 5
    });

    // Can only have 2 clusters max with 3 samples
    expect(result.evaluations).toHaveLength(1);
    expect(result.optimal.k).toBe(2);
  });

  it("should throw on invalid parameters", async () => {
    const data = [[1, 2], [2, 3]];

    await expect(
      find_optimal_clusters(data, { min_clusters: 1 })
    ).rejects.toThrow('minClusters must be at least 2');

    await expect(
      find_optimal_clusters(data, { min_clusters: 3, max_clusters: 2 })
    ).rejects.toThrow('maxClusters must be greater than or equal to minClusters');

    // Use more samples to avoid the "not enough samples" error
    const more_data = [[1, 2], [2, 3], [10, 11], [11, 12]];
    await expect(
      find_optimal_clusters(more_data, { algorithm: 'invalid' as any })
    ).rejects.toThrow('Unknown algorithm: invalid');
  });

  it("should pass algorithm parameters", async () => {
    const data = [
      [1, 2], [1.5, 1.8], [5, 8], [8, 8], [1, 0.6], [9, 11]
    ];

    const result = await find_optimal_clusters(data, {
      algorithm: 'kmeans',
      algorithm_params: { n_init: 5, max_iter: 100 },
      max_clusters: 3
    });

    expect(result.optimal.k).toBeGreaterThanOrEqual(2);
    expect(result.optimal.k).toBeLessThanOrEqual(3);
  });

  describe("Normalized scoring (AC#1)", () => {
    it("should produce combined scores in [0, 1] range", async () => {
      const data = [
        [1, 2], [1.5, 1.8], [5, 8], [8, 8], [1, 0.6], [9, 11],
        [1.2, 1.9], [1.4, 1.7], [5.2, 8.1], [8.1, 8.2], [0.9, 0.5], [9.1, 11.2]
      ];

      const result = await find_optimal_clusters(data, { max_clusters: 4 });

      for (const e of result.evaluations) {
        expect(e.combined_score).toBeGreaterThanOrEqual(0);
        expect(e.combined_score).toBeLessThanOrEqual(1);
      }
    });

    it("should not let calinski-harabasz dominate scoring", async () => {
      const data = [
        [1, 2], [1.5, 1.8], [5, 8], [8, 8], [1, 0.6], [9, 11],
        [1.2, 1.9], [1.4, 1.7], [5.2, 8.1], [8.1, 8.2], [0.9, 0.5], [9.1, 11.2]
      ];

      const result = await find_optimal_clusters(data, { max_clusters: 4 });

      // Verify that the combined score is NOT dominated by CH
      // (old behavior: score = silhouette + calinskiHarabasz - daviesBouldin)
      // New behavior: all normalized to [0,1] and averaged
      for (const e of result.evaluations) {
        expect(e.combined_score).not.toBe(e.calinski_harabasz);
      }
    });

    it("should handle single evaluation (min === max)", async () => {
      const data = [[1, 2], [10, 11], [5, 6]];

      const result = await find_optimal_clusters(data, {
        min_clusters: 2,
        max_clusters: 2
      });

      expect(result.evaluations).toHaveLength(1);
      expect(isFinite(result.optimal.combined_score)).toBe(true);
      expect(isNaN(result.optimal.combined_score)).toBe(false);
    });

    it("should still pass raw values to custom scoringFunction", async () => {
      const data = [[1, 2], [2, 3], [10, 11], [11, 12]];

      const result = await find_optimal_clusters(data, {
        max_clusters: 3,
        scoring_function: (evaluation) => evaluation.calinski_harabasz
      });

      // Custom scorer gets raw (un-normalized) CH values
      expect(result.optimal.combined_score).toBe(result.optimal.calinski_harabasz);
    });
  });

  describe("Silhouette-only method (AC#3)", () => {
    it("should select k with highest silhouette", async () => {
      const data = [
        [1, 2], [1.5, 1.8], [5, 8], [8, 8], [1, 0.6], [9, 11],
        [1.2, 1.9], [1.4, 1.7], [5.2, 8.1], [8.1, 8.2], [0.9, 0.5], [9.1, 11.2]
      ];

      const result = await find_optimal_clusters(data, {
        method: 'silhouette',
        max_clusters: 5
      });

      // Combined score should equal silhouette
      expect(result.optimal.combined_score).toBe(result.optimal.silhouette);

      // Should be sorted by silhouette (descending)
      for (let i = 1; i < result.evaluations.length; i++) {
        expect(result.evaluations[i - 1].combined_score).toBeGreaterThanOrEqual(
          result.evaluations[i].combined_score
        );
      }
    });

    it("should not compute DB or CH when using silhouette method", async () => {
      const data = [[1, 2], [2, 3], [10, 11], [11, 12]];

      const result = await find_optimal_clusters(data, {
        method: 'silhouette',
        max_clusters: 3
      });

      // When silhouette method is used, DB and CH should be defaults
      for (const e of result.evaluations) {
        expect(e.davies_bouldin).toBe(Infinity);
        expect(e.calinski_harabasz).toBe(0);
      }
    });
  });

  describe("Elbow method (AC#2)", () => {
    it("should populate wss field", async () => {
      const data = [
        [1, 2], [1.5, 1.8], [5, 8], [8, 8], [1, 0.6], [9, 11]
      ];

      const result = await find_optimal_clusters(data, {
        method: 'elbow',
        max_clusters: 4
      });

      for (const e of result.evaluations) {
        expect(e.wss).toBeDefined();
        expect(e.wss).toBeGreaterThanOrEqual(0);
      }
    });

    it("should detect a reasonable knee for well-separated data", async () => {
      const { X } = make_blobs({
        n_samples: 60,
        n_features: 2,
        centers: 3,
        cluster_std: 0.5,
        random_state: 42
      });

      try {
        const result = await find_optimal_clusters(X, {
          method: 'elbow',
          min_clusters: 2,
          max_clusters: 6
        });

        // The knee should be around k=3
        expect(result.optimal.k).toBeGreaterThanOrEqual(2);
        expect(result.optimal.k).toBeLessThanOrEqual(5);
      } finally {
        X.dispose();
      }
    });

    it("should use KMeans inertia when algorithm is kmeans", async () => {
      const data = [[1, 2], [2, 3], [10, 11], [11, 12]];

      const result = await find_optimal_clusters(data, {
        method: 'elbow',
        algorithm: 'kmeans',
        max_clusters: 3
      });

      // WSS should be populated and positive
      for (const e of result.evaluations) {
        expect(e.wss).toBeDefined();
        expect(e.wss!).toBeGreaterThan(0);
      }
    });
  });

  it("should override method when scoringFunction is provided", async () => {
    const data = [[1, 2], [2, 3], [10, 11], [11, 12]];

    const result = await find_optimal_clusters(data, {
      method: 'silhouette',
      max_clusters: 3,
      scoring_function: (evaluation) => evaluation.silhouette * 100
    });

    // Custom scorer should override method
    expect(result.optimal.combined_score).toBe(result.optimal.silhouette * 100);
  });
});