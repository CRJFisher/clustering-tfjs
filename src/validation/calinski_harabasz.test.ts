import { describe, it, expect, beforeEach, afterEach } from "@jest/globals";
import * as tf from "../../test_support/tensorflow_helper";
import { calinski_harabasz, calinski_harabasz_efficient } from "./calinski_harabasz";
import { make_random_stream } from "../random";

describe("Calinski-Harabasz Score", () => {
  beforeEach(() => {
    tf.engine().startScope();
  });

  afterEach(() => {
    tf.engine().endScope();
  });

  describe("Basic functionality", () => {
    it("should compute correct score for well-separated 2D clusters", () => {
      // Create well-separated clusters
      const X = [
        // Cluster 0: centered around (0, 0)
        [0.1, 0.1], [0.2, 0.0], [-0.1, 0.2], [0.0, -0.1],
        // Cluster 1: centered around (5, 5)
        [5.1, 5.0], [4.9, 5.1], [5.0, 4.9], [5.2, 5.2],
        // Cluster 2: centered around (5, 0)
        [5.0, 0.1], [5.1, -0.1], [4.9, 0.0], [5.0, 0.2]
      ];
      const labels = [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2];
      
      const score = calinski_harabasz(X, labels);
      
      // Well-separated clusters should have high score
      expect(score).toBeGreaterThan(100);
    });

    it("should compute same score with tensor inputs", () => {
      const X = [[1, 2], [1.5, 1.8], [5, 8], [8, 8], [1, 0.6], [9, 11]];
      const labels = [0, 0, 1, 1, 0, 1];
      
      const score_array = calinski_harabasz(X, labels);
      
      const X_tensor = tf.tensor2d(X);
      const labels_tensor = tf.tensor1d(labels);
      const score_tensor = calinski_harabasz(X_tensor, labels_tensor);
      
      expect(score_tensor).toBeCloseTo(score_array, 5);
      
      X_tensor.dispose();
      labels_tensor.dispose();
    });

    it("should return same result for efficient version", () => {
      const X = [
        [0, 0], [0.5, 0.5], [0.5, -0.5], [-0.5, 0.5],
        [10, 10], [10.5, 10.5], [10.5, 9.5], [9.5, 10.5]
      ];
      const labels = [0, 0, 0, 0, 1, 1, 1, 1];
      
      const score1 = calinski_harabasz(X, labels);
      const score2 = calinski_harabasz_efficient(X, labels);
      
      expect(score1).toBeCloseTo(score2, 10);
    });
  });

  describe("Edge cases", () => {
    it("should throw error for single cluster", () => {
      const X = [[1, 2], [3, 4], [5, 6]];
      const labels = [0, 0, 0];
      
      expect(() => calinski_harabasz(X, labels)).toThrow(
        "Calinski-Harabasz score requires at least 2 clusters"
      );
    });

    it("should throw error when k >= n_samples", () => {
      const X = [[1, 2], [3, 4], [5, 6]];
      const labels = [0, 1, 2];
      
      expect(() => calinski_harabasz(X, labels)).toThrow(
        "Number of clusters must be less than number of samples"
      );
    });

    it("should handle clusters of different sizes", () => {
      const X = [
        // Large cluster 0
        [0, 0], [0.1, 0.1], [0.2, 0], [-0.1, 0.1], [0, -0.1],
        // Small cluster 1
        [5, 5], [5.1, 5.1],
        // Medium cluster 2
        [0, 5], [0.1, 5], [-0.1, 5]
      ];
      const labels = [0, 0, 0, 0, 0, 1, 1, 2, 2, 2];
      
      const score = calinski_harabasz(X, labels);
      expect(score).toBeGreaterThan(0);
      expect(isFinite(score)).toBe(true);
    });
  });

  describe("Known values validation", () => {
    it("should match sklearn example values", () => {
      // Example from sklearn documentation
      const X = [
        [1, 2], [1, 4], [1, 0],
        [4, 2], [4, 4], [4, 0]
      ];
      const labels = [0, 0, 0, 1, 1, 1];
      
      const score = calinski_harabasz(X, labels);
      
      // sklearn gives 3.375 for this example
      expect(score).toBeCloseTo(3.375, 3);
    });

    it("should compute correct score for iris-like data", () => {
      // Simplified iris-like dataset with 3 features, 3 clusters
      const X = [
        // Cluster 0 (like setosa)
        [5.1, 3.5, 1.4], [4.9, 3.0, 1.4], [4.7, 3.2, 1.3], [4.6, 3.1, 1.5],
        // Cluster 1 (like versicolor)  
        [7.0, 3.2, 4.7], [6.4, 3.2, 4.5], [6.9, 3.1, 4.9], [5.5, 2.3, 4.0],
        // Cluster 2 (like virginica)
        [6.3, 3.3, 6.0], [5.8, 2.7, 5.1], [7.1, 3.0, 5.9], [6.3, 2.9, 5.6]
      ];
      const labels = [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2];

      const score = calinski_harabasz(X, labels);

      // sklearn.metrics.calinski_harabasz_score on this input.
      expect(score).toBeCloseTo(48.02278037383177, 4);
    });
  });

  describe("Numerical stability", () => {
    it("should handle very small distances", () => {
      const X = [
        [0, 0], [0.0001, 0.0001], [0.0002, 0],
        [1, 1], [1.0001, 1.0001], [1.0002, 1]
      ];
      const labels = [0, 0, 0, 1, 1, 1];
      
      const score = calinski_harabasz(X, labels);
      expect(isFinite(score)).toBe(true);
      expect(score).toBeGreaterThan(0);
    });

    it("should handle large feature values", () => {
      const X = [
        [1000, 2000], [1001, 2001], [999, 1999],
        [5000, 6000], [5001, 6001], [4999, 5999]
      ];
      const labels = [0, 0, 0, 1, 1, 1];
      
      const score = calinski_harabasz(X, labels);
      expect(isFinite(score)).toBe(true);
      expect(score).toBeGreaterThan(0);
    });
  });

  describe("Performance", () => {
    it("should handle moderately large datasets efficiently", () => {
      // Create 1000 samples in 3 clusters
      const n = 1000;
      const X: number[][] = [];
      const labels: number[] = [];
      
      // Generate 3 clusters
      const rng = make_random_stream(42);
      for (let i = 0; i < n; i++) {
        const cluster = Math.floor(i / (n / 3));
        labels.push(cluster);

        // Add noise to cluster centers
        const noise = () => (rng.rand() - 0.5) * 0.5;
        if (cluster === 0) {
          X.push([0 + noise(), 0 + noise()]);
        } else if (cluster === 1) {
          X.push([5 + noise(), 5 + noise()]);
        } else {
          X.push([5 + noise(), 0 + noise()]);
        }
      }
      
      const start = Date.now();
      const score = calinski_harabasz_efficient(X, labels);
      const elapsed = Date.now() - start;
      
      expect(score).toBeGreaterThan(100); // Well-separated clusters
      expect(elapsed).toBeLessThan(1000); // Should complete in < 1 second
    });
  });

  describe("Labels length validation", () => {
    it("should throw when labels length mismatches data rows", () => {
      const X = [[1, 2], [3, 4], [5, 6]];
      const labels = [0, 1];

      expect(() => calinski_harabasz(X, labels)).toThrow(
        "Labels length (2) does not match data rows (3)"
      );
    });

    it("should throw for efficient version", () => {
      const X = [[1, 2], [3, 4], [5, 6]];
      const labels = [0, 1];

      expect(() => calinski_harabasz_efficient(X, labels)).toThrow(
        "Labels length (2) does not match data rows (3)"
      );
    });
  });
});
describe("Calinski-Harabasz – noise (-1) awareness", () => {
  const two_clusters = [
    [0, 0],
    [0.1, 0],
    [5, 5],
    [5.1, 5],
  ];

  it("excludes noise so score equals the noise-free score", () => {
    const base = calinski_harabasz(two_clusters, [0, 0, 1, 1]);
    const with_noise = calinski_harabasz(
      [...two_clusters, [100, 100], [-100, -100]],
      [0, 0, 1, 1, -1, -1],
    );
    expect(with_noise).toBeCloseTo(base, 4);
  });

  it("returns defined 0 when every label is noise", () => {
    expect(calinski_harabasz(two_clusters, [-1, -1, -1, -1])).toBe(0);
    expect(calinski_harabasz_efficient(two_clusters, [-1, -1, -1, -1])).toBe(0);
  });

  it("returns defined 0 for a single cluster plus noise", () => {
    expect(calinski_harabasz(two_clusters, [0, 0, 0, -1])).toBe(0);
    expect(calinski_harabasz_efficient(two_clusters, [0, 0, 0, -1])).toBe(0);
  });

  it("returns defined 0 when noise leaves k >= remaining samples", () => {
    // After dropping the two noise rows, two samples remain in two distinct
    // clusters (k === n), the degenerate case — defined as 0, not a throw.
    const labels = [0, 1, -1, -1];
    expect(calinski_harabasz(two_clusters, labels)).toBe(0);
    expect(calinski_harabasz_efficient(two_clusters, labels)).toBe(0);
  });
});
