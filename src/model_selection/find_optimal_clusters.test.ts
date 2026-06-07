import { find_optimal_clusters } from "./find_optimal_clusters";
import { make_blobs } from "../datasets/synthetic";
import * as tf from "../../test_support/tensorflow_helper";

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
    const { X } = make_blobs({
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

  it("SOM two-phase path produces exactly k clusters for each k", async () => {
    // Three well-separated blobs.
    const { X } = make_blobs({
      n_samples: 60,
      n_features: 2,
      centers: 3,
      cluster_std: 0.4,
      random_state: 42,
    });

    try {
      const result = await find_optimal_clusters(X, {
        algorithm: 'som',
        min_clusters: 2,
        max_clusters: 5,
        algorithm_params: { num_epochs: 20, random_state: 42 },
      });

      // One evaluation per k in [2, 5].
      expect(result.evaluations.map((e) => e.k).sort()).toEqual([2, 3, 4, 5]);

      // The core fix: the output cluster count is driven by k (via two-phase
      // grouping of neuron weights), NOT by the SOM grid size. Each evaluation
      // yields at most k labels — never the grid's neuron count. (A macro-
      // cluster of neurons with no assigned samples can leave fewer than k
      // distinct labels, which is expected.)
      for (const evaluation of result.evaluations) {
        const unique = new Set(evaluation.labels);
        expect(unique.size).toBeGreaterThanOrEqual(2);
        expect(unique.size).toBeLessThanOrEqual(evaluation.k);
        expect(evaluation.labels).toHaveLength(60);
      }

      // The sweep should recover the 3 underlying blobs exactly.
      const optimal_unique = new Set(result.optimal.labels);
      expect(result.optimal.k).toBe(3);
      expect(optimal_unique.size).toBe(3);
    } finally {
      X.dispose();
    }
  });

  it("SOM degenerate collapse (<2 distinct labels) yields worst-case metrics without crashing", async () => {
    // All-identical points → every sample shares one BMU → the two-phase
    // mapping collapses to a single label for every k. The metric functions
    // require >=2 clusters, so this exercises the degenerate-label guard.
    const data = Array.from({ length: 20 }, () => [1, 1]);

    const result = await find_optimal_clusters(data, {
      algorithm: 'som',
      min_clusters: 2,
      max_clusters: 4,
      metrics: ['silhouette', 'davies_bouldin', 'calinski_harabasz'],
      algorithm_params: { num_epochs: 5, random_state: 42 },
    });

    for (const e of result.evaluations) {
      expect(new Set(e.labels).size).toBe(1);
      expect(e.silhouette).toBe(-1);
      expect(e.davies_bouldin).toBe(Infinity);
      expect(e.calinski_harabasz).toBe(0);
    }
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
    ).rejects.toThrow('min_clusters must be at least 2');

    await expect(
      find_optimal_clusters(data, { min_clusters: 3, max_clusters: 2 })
    ).rejects.toThrow('max_clusters must be greater than or equal to min_clusters');

    // Use more samples to avoid the "not enough samples" error
    const more_data = [[1, 2], [2, 3], [10, 11], [11, 12]];
    await expect(
      // @ts-expect-error - invalid algorithm value; testing runtime validation
      find_optimal_clusters(more_data, { algorithm: 'invalid' })
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
      // (old behavior: score = silhouette + calinski_harabasz - davies_bouldin)
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