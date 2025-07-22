import { describe, it, expect } from "@jest/globals";
import * as tf from "@tensorflow/tfjs-node";
import { silhouetteScore, silhouetteScoreSubset } from "../../src/validation/silhouette";

describe("Silhouette Score", () => {
  afterEach(() => {
    // Clean up any remaining tensors
    tf.engine().startScope();
    tf.engine().endScope();
  });

  describe("Basic functionality", () => {
    it("should compute high score for well-separated 2D clusters", () => {
      // Create well-separated clusters
      const X = [
        // Cluster 0: centered around (0, 0)
        [0, 0], [0.1, 0.1], [0.2, 0], [-0.1, 0.1],
        // Cluster 1: centered around (10, 10)
        [10, 10], [10.1, 10.1], [9.9, 9.9], [10.1, 9.9]
      ];
      const labels = [0, 0, 0, 0, 1, 1, 1, 1];
      
      const score = silhouetteScore(X, labels);
      
      // Well-separated clusters should have high score
      expect(score).toBeGreaterThan(0.8);
      expect(score).toBeLessThanOrEqual(1.0);
    });

    it("should compute low score for overlapping clusters", () => {
      // Create overlapping clusters
      const X = [
        // Overlapping clusters
        [0, 0], [0.5, 0.5], [1, 1], [1.5, 1.5],
        [0.5, 0], [1, 0.5], [1.5, 1], [2, 1.5]
      ];
      const labels = [0, 0, 0, 0, 1, 1, 1, 1];
      
      const score = silhouetteScore(X, labels);
      
      // Overlapping clusters should have lower score
      expect(score).toBeLessThan(0.5);
      expect(score).toBeGreaterThan(-1.0);
    });

    it("should compute negative score for misclassified points", () => {
      // Create clusters with misclassified points
      const X = [
        // Cluster 0 points
        [0, 0], [0.1, 0.1], [0.2, 0],
        // Misclassified point (labeled as 0 but closer to cluster 1)
        [9.8, 9.8],
        // Cluster 1 points
        [10, 10], [10.1, 10.1], [9.9, 9.9],
        // Misclassified point (labeled as 1 but closer to cluster 0)
        [0.3, 0.3]
      ];
      const labels = [0, 0, 0, 0, 1, 1, 1, 1];
      
      const score = silhouetteScore(X, labels);
      
      // Misclassified points should lower the score
      expect(score).toBeLessThan(0.5);
    });

    it("should compute same score with tensor inputs", () => {
      const X = [[1, 2], [1.5, 1.8], [5, 8], [8, 8], [1, 0.6], [9, 11]];
      const labels = [0, 0, 1, 1, 0, 1];
      
      const scoreArray = silhouetteScore(X, labels);
      
      const XTensor = tf.tensor2d(X);
      const labelsTensor = tf.tensor1d(labels);
      const scoreTensor = silhouetteScore(XTensor, labelsTensor);
      
      expect(scoreTensor).toBeCloseTo(scoreArray, 5);
      
      XTensor.dispose();
      labelsTensor.dispose();
    });
  });

  describe("Edge cases", () => {
    it("should throw error for single cluster", () => {
      const X = [[1, 2], [3, 4], [5, 6]];
      const labels = [0, 0, 0];
      
      expect(() => silhouetteScore(X, labels)).toThrow(
        "Silhouette score requires at least 2 clusters"
      );
    });

    it("should handle single-point clusters", () => {
      const X = [
        [0, 0], [0.1, 0.1], // Cluster 0 (2 points)
        [5, 5],             // Cluster 1 (1 point) - should get score 0
        [10, 10], [10.1, 10.1] // Cluster 2 (2 points)
      ];
      const labels = [0, 0, 1, 2, 2];
      
      const score = silhouetteScore(X, labels);
      expect(isFinite(score)).toBe(true);
      expect(score).toBeGreaterThanOrEqual(-1);
      expect(score).toBeLessThanOrEqual(1);
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
      
      const score = silhouetteScore(X, labels);
      expect(isFinite(score)).toBe(true);
      expect(score).toBeGreaterThan(0);
    });
  });

  describe("Known values validation", () => {
    it("should compute perfect score for perfectly separated clusters", () => {
      // Two clusters far apart
      const X = [
        [0, 0], [0, 0], [0, 0],    // Cluster 0 (all same point)
        [100, 100], [100, 100], [100, 100]  // Cluster 1 (all same point, far away)
      ];
      const labels = [0, 0, 0, 1, 1, 1];
      
      const score = silhouetteScore(X, labels);
      
      // Should be close to 1 (perfect separation)
      expect(score).toBeCloseTo(1.0, 2);
    });

    it("should compute zero score for points on decision boundary", () => {
      // Points equally distant from both clusters
      const X = [
        // Cluster 0
        [0, 0], [0, 2],
        // Points on boundary (at x=1)
        [1, 0], [1, 2],
        // Cluster 1
        [2, 0], [2, 2]
      ];
      const labels = [0, 0, 0, 0, 1, 1];
      
      const score = silhouetteScore(X, labels);
      
      // Should be close to 0 (on boundary)
      expect(Math.abs(score)).toBeLessThan(0.2);
    });
  });

  describe("Subset computation", () => {
    it("should compute same score for subset as full computation", () => {
      const X = [
        [0, 0], [0.1, 0.1], [0.2, 0], [-0.1, 0.1],
        [5, 5], [5.1, 5.1], [4.9, 4.9], [5.1, 4.9]
      ];
      const labels = [0, 0, 0, 0, 1, 1, 1, 1];
      
      // Compute full score
      const fullScore = silhouetteScore(X, labels);
      
      // Compute subset score for all indices
      const allIndices = Array.from({ length: 8 }, (_, i) => i);
      const subsetScore = silhouetteScoreSubset(X, labels, allIndices);
      
      expect(subsetScore).toBeCloseTo(fullScore, 5);
    });

    it("should handle partial subset correctly", () => {
      const X = [
        [0, 0], [0.1, 0.1], [0.2, 0], [-0.1, 0.1],
        [5, 5], [5.1, 5.1], [4.9, 4.9], [5.1, 4.9]
      ];
      const labels = [0, 0, 0, 0, 1, 1, 1, 1];
      
      // Compute score for subset of indices
      const subsetIndices = [0, 2, 4, 6]; // Sample from each cluster
      const score = silhouetteScoreSubset(X, labels, subsetIndices);
      
      expect(score).toBeGreaterThan(0.5); // Still well-separated
      expect(score).toBeLessThanOrEqual(1.0);
    });
  });

  describe("Numerical stability", () => {
    it("should handle identical points in same cluster", () => {
      const X = [
        [1, 1], [1, 1], [1, 1],    // Identical points in cluster 0
        [5, 5], [5.1, 5.1], [4.9, 4.9]  // Cluster 1
      ];
      const labels = [0, 0, 0, 1, 1, 1];
      
      const score = silhouetteScore(X, labels);
      expect(isFinite(score)).toBe(true);
      expect(score).toBeGreaterThan(0);
    });

    it("should handle very small distances", () => {
      const X = [
        [0, 0], [0.0001, 0.0001], [0.0002, 0],
        [1, 1], [1.0001, 1.0001], [1.0002, 1]
      ];
      const labels = [0, 0, 0, 1, 1, 1];
      
      const score = silhouetteScore(X, labels);
      expect(isFinite(score)).toBe(true);
      expect(score).toBeGreaterThan(0);
    });
  });

  describe("Performance", () => {
    it("subset computation should be faster for large datasets", () => {
      // Create 500 samples in 2 clusters
      const n = 500;
      const X: number[][] = [];
      const labels: number[] = [];
      
      for (let i = 0; i < n; i++) {
        const cluster = i < n/2 ? 0 : 1;
        labels.push(cluster);
        
        const noise = () => (Math.random() - 0.5) * 0.5;
        if (cluster === 0) {
          X.push([0 + noise(), 0 + noise()]);
        } else {
          X.push([5 + noise(), 5 + noise()]);
        }
      }
      
      // Time full computation
      const fullStart = Date.now();
      const fullScore = silhouetteScore(X, labels);
      const fullTime = Date.now() - fullStart;
      
      // Time subset computation (10% of samples)
      const subsetSize = Math.floor(n * 0.1);
      const subsetIndices = Array.from({ length: subsetSize }, 
        (_, i) => Math.floor(i * n / subsetSize));
      
      const subsetStart = Date.now();
      const subsetScore = silhouetteScoreSubset(X, labels, subsetIndices);
      const subsetTime = Date.now() - subsetStart;
      
      // Subset should be faster (but not necessarily 3x faster due to overhead)
      expect(subsetTime).toBeLessThan(fullTime);
      
      // Scores should be similar
      expect(Math.abs(fullScore - subsetScore)).toBeLessThan(0.1);
    });
  });
});