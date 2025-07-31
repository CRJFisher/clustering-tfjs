import { describe, it, expect } from "@jest/globals";
import * as tf from "../tensorflow-helper";
import { calinskiHarabasz, calinskiHarabaszEfficient } from "../../src/validation/calinski_harabasz";

describe("Calinski-Harabasz Score", () => {
  afterEach(() => {
    // Clean up any remaining tensors
    tf.engine().startScope();
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
      
      const score = calinskiHarabasz(X, labels);
      
      // Well-separated clusters should have high score
      expect(score).toBeGreaterThan(100);
    });

    it("should compute same score with tensor inputs", () => {
      const X = [[1, 2], [1.5, 1.8], [5, 8], [8, 8], [1, 0.6], [9, 11]];
      const labels = [0, 0, 1, 1, 0, 1];
      
      const scoreArray = calinskiHarabasz(X, labels);
      
      const XTensor = tf.tensor2d(X);
      const labelsTensor = tf.tensor1d(labels);
      const scoreTensor = calinskiHarabasz(XTensor, labelsTensor);
      
      expect(scoreTensor).toBeCloseTo(scoreArray, 5);
      
      XTensor.dispose();
      labelsTensor.dispose();
    });

    it("should return same result for efficient version", () => {
      const X = [
        [0, 0], [0.5, 0.5], [0.5, -0.5], [-0.5, 0.5],
        [10, 10], [10.5, 10.5], [10.5, 9.5], [9.5, 10.5]
      ];
      const labels = [0, 0, 0, 0, 1, 1, 1, 1];
      
      const score1 = calinskiHarabasz(X, labels);
      const score2 = calinskiHarabaszEfficient(X, labels);
      
      expect(score1).toBeCloseTo(score2, 10);
    });
  });

  describe("Edge cases", () => {
    it("should throw error for single cluster", () => {
      const X = [[1, 2], [3, 4], [5, 6]];
      const labels = [0, 0, 0];
      
      expect(() => calinskiHarabasz(X, labels)).toThrow(
        "Calinski-Harabasz score requires at least 2 clusters"
      );
    });

    it("should throw error when k >= n_samples", () => {
      const X = [[1, 2], [3, 4], [5, 6]];
      const labels = [0, 1, 2];
      
      expect(() => calinskiHarabasz(X, labels)).toThrow(
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
      
      const score = calinskiHarabasz(X, labels);
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
      
      const score = calinskiHarabasz(X, labels);
      
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
      
      const score = calinskiHarabasz(X, labels);
      
      // Should have reasonably high score for well-separated clusters
      expect(score).toBeGreaterThan(20);
    });
  });

  describe("Numerical stability", () => {
    it("should handle very small distances", () => {
      const X = [
        [0, 0], [0.0001, 0.0001], [0.0002, 0],
        [1, 1], [1.0001, 1.0001], [1.0002, 1]
      ];
      const labels = [0, 0, 0, 1, 1, 1];
      
      const score = calinskiHarabasz(X, labels);
      expect(isFinite(score)).toBe(true);
      expect(score).toBeGreaterThan(0);
    });

    it("should handle large feature values", () => {
      const X = [
        [1000, 2000], [1001, 2001], [999, 1999],
        [5000, 6000], [5001, 6001], [4999, 5999]
      ];
      const labels = [0, 0, 0, 1, 1, 1];
      
      const score = calinskiHarabasz(X, labels);
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
      for (let i = 0; i < n; i++) {
        const cluster = Math.floor(i / (n / 3));
        labels.push(cluster);
        
        // Add noise to cluster centers
        const noise = () => (Math.random() - 0.5) * 0.5;
        if (cluster === 0) {
          X.push([0 + noise(), 0 + noise()]);
        } else if (cluster === 1) {
          X.push([5 + noise(), 5 + noise()]);
        } else {
          X.push([5 + noise(), 0 + noise()]);
        }
      }
      
      const start = Date.now();
      const score = calinskiHarabaszEfficient(X, labels);
      const elapsed = Date.now() - start;
      
      expect(score).toBeGreaterThan(100); // Well-separated clusters
      expect(elapsed).toBeLessThan(1000); // Should complete in < 1 second
    });
  });
});