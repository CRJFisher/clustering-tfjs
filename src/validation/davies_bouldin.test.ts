import * as tf from "../../test_support/tensorflow_helper";
import { davies_bouldin, davies_bouldin_efficient } from "./davies_bouldin";
import { make_random_stream } from "../random";

describe("davies_bouldin", () => {
  beforeEach(() => {
    tf.engine().startScope();
  });

  afterEach(() => {
    tf.engine().endScope();
  });

  describe("Basic functionality", () => {
    it("should compute low score for well-separated 2D clusters", () => {
      // Create well-separated clusters
      const X = [
        // Cluster 0: centered around (0, 0)
        [0, 0], [0.1, 0.1], [0.2, 0], [-0.1, 0.1],
        // Cluster 1: centered around (10, 10)
        [10, 10], [10.1, 10.1], [9.9, 9.9], [10.1, 9.9],
        // Cluster 2: centered around (10, 0)
        [10, 0], [10.1, 0.1], [9.9, 0], [10, -0.1]
      ];
      const labels = [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2];
      
      const score = davies_bouldin(X, labels);
      
      // Well-separated clusters should have low score
      expect(score).toBeLessThan(0.5);
      expect(score).toBeGreaterThan(0);
    });

    it("should compute high score for overlapping clusters", () => {
      // Create overlapping clusters
      const X = [
        // Overlapping clusters
        [0, 0], [0.5, 0.5], [1, 1], [1.5, 1.5],
        [0.5, 0], [1, 0.5], [1.5, 1], [2, 1.5]
      ];
      const labels = [0, 0, 0, 0, 1, 1, 1, 1];
      
      const score = davies_bouldin(X, labels);
      
      // Overlapping clusters should have higher score
      expect(score).toBeGreaterThan(0.5);
    });

    it("should compute same score with tensor inputs", () => {
      const X = [[1, 2], [1.5, 1.8], [5, 8], [8, 8], [1, 0.6], [9, 11]];
      const labels = [0, 0, 1, 1, 0, 1];
      
      const score_array = davies_bouldin(X, labels);
      
      const X_tensor = tf.tensor2d(X);
      const labels_tensor = tf.tensor1d(labels);
      const score_tensor = davies_bouldin(X_tensor, labels_tensor);
      
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
      
      const score1 = davies_bouldin(X, labels);
      const score2 = davies_bouldin_efficient(X, labels);
      
      expect(score1).toBeCloseTo(score2, 10);
    });
  });

  describe("Edge cases", () => {
    it("should throw error for single cluster", () => {
      const X = [[1, 2], [3, 4], [5, 6]];
      const labels = [0, 0, 0];
      
      expect(() => davies_bouldin(X, labels)).toThrow(
        "Davies-Bouldin score requires at least 2 clusters"
      );
    });

    it("should handle single-point clusters", () => {
      const X = [
        [0, 0], [0.1, 0.1], // Cluster 0 (2 points)
        [5, 5],             // Cluster 1 (1 point)
        [10, 10], [10.1, 10.1] // Cluster 2 (2 points)
      ];
      const labels = [0, 0, 1, 2, 2];
      
      const score = davies_bouldin(X, labels);
      expect(isFinite(score)).toBe(true);
      expect(score).toBeGreaterThan(0);
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
      
      const score = davies_bouldin(X, labels);
      expect(isFinite(score)).toBe(true);
      expect(score).toBeGreaterThan(0);
    });

    it("should return Infinity for coincident centroids with nonzero dispersions", () => {
      // Two clusters with same centroid but different dispersions
      const X = [
        [-1, 0], [1, 0], [0, -1], [0, 1],
        [-0.1, 0], [0.1, 0], [0, -0.1], [0, 0.1]
      ];
      const labels = [0, 0, 0, 0, 1, 1, 1, 1];

      const score = davies_bouldin(X, labels);
      expect(score).toBe(Infinity);
    });

    it("should return 0 for coincident centroids with zero dispersions", () => {
      const X = [
        [1, 1], [1, 1], [1, 1],
        [1, 1], [1, 1], [1, 1],
      ];
      const labels = [0, 0, 0, 1, 1, 1];

      const score = davies_bouldin(X, labels);
      expect(score).toBe(0);
      expect(isFinite(score)).toBe(true);
    });

    it("should return 0 for coincident centroids with zero dispersions (efficient)", () => {
      const X = [
        [1, 1], [1, 1], [1, 1],
        [1, 1], [1, 1], [1, 1],
      ];
      const labels = [0, 0, 0, 1, 1, 1];

      const score = davies_bouldin_efficient(X, labels);
      expect(score).toBe(0);
      expect(isFinite(score)).toBe(true);
    });

    it("should handle mixed: some coincident, some separate centroids", () => {
      const X = [
        [0, 0], [0, 0],
        [0, 0], [0, 0],
        [10, 10], [10.1, 10.1],
      ];
      const labels = [0, 0, 1, 1, 2, 2];

      const score = davies_bouldin(X, labels);
      expect(isFinite(score)).toBe(true);
      expect(score).toBeGreaterThanOrEqual(0);
    });
  });

  describe("Labels length validation", () => {
    it("should throw when labels length mismatches data rows", () => {
      const X = [[1, 2], [3, 4], [5, 6]];
      const labels = [0, 1];

      expect(() => davies_bouldin(X, labels)).toThrow(
        "Labels length (2) does not match data rows (3)"
      );
    });

    it("should throw for efficient version", () => {
      const X = [[1, 2], [3, 4], [5, 6]];
      const labels = [0, 1];

      expect(() => davies_bouldin_efficient(X, labels)).toThrow(
        "Labels length (2) does not match data rows (3)"
      );
    });
  });

  describe("Known values validation", () => {
    it("should match expected value for simple 2-cluster example", () => {
      // Simple well-separated 2-cluster example
      const X = [
        [0, 0], [0, 1], [1, 0], [1, 1],     // Cluster 0 (square at origin)
        [5, 5], [5, 6], [6, 5], [6, 6]      // Cluster 1 (square at (5,5))
      ];
      const labels = [0, 0, 0, 0, 1, 1, 1, 1];
      
      const score = davies_bouldin(X, labels);
      
      // Manual calculation:
      // Centroid 0: (0.5, 0.5), Centroid 1: (5.5, 5.5)
      // s_0 ≈ 0.707, s_1 ≈ 0.707
      // d_01 ≈ 7.07
      // R_01 = (0.707 + 0.707) / 7.07 ≈ 0.2
      // DB = 0.2
      
      expect(score).toBeCloseTo(0.2, 1);
    });

    it("should give lower score for better separated clusters", () => {
      // Test 1: Less separated clusters
      const X1 = [
        [0, 0], [1, 0], [0, 1],
        [3, 3], [4, 3], [3, 4]
      ];
      const labels1 = [0, 0, 0, 1, 1, 1];
      const score1 = davies_bouldin(X1, labels1);
      
      // Test 2: Well separated clusters
      const X2 = [
        [0, 0], [1, 0], [0, 1],
        [10, 10], [11, 10], [10, 11]
      ];
      const labels2 = [0, 0, 0, 1, 1, 1];
      const score2 = davies_bouldin(X2, labels2);
      
      // Better separated clusters should have lower score
      expect(score2).toBeLessThan(score1);
    });
  });

  describe("Numerical stability", () => {
    it("should handle very small distances", () => {
      const X = [
        [0, 0], [0.0001, 0.0001], [0.0002, 0],
        [1, 1], [1.0001, 1.0001], [1.0002, 1]
      ];
      const labels = [0, 0, 0, 1, 1, 1];
      
      const score = davies_bouldin(X, labels);
      expect(isFinite(score)).toBe(true);
      expect(score).toBeGreaterThan(0);
    });

    it("should handle large feature values", () => {
      const X = [
        [1000, 2000], [1001, 2001], [999, 1999],
        [5000, 6000], [5001, 6001], [4999, 5999]
      ];
      const labels = [0, 0, 0, 1, 1, 1];
      
      const score = davies_bouldin(X, labels);
      expect(isFinite(score)).toBe(true);
      expect(score).toBeGreaterThan(0);
      expect(score).toBeLessThan(1); // Should still be low for well-separated clusters
    });
  });

  describe("Performance", () => {
    it("should handle moderately large datasets efficiently", () => {
      // Create 1000 samples in 3 clusters
      const n = 1000;
      const X: number[][] = [];
      const labels: number[] = [];
      
      // Generate 3 well-separated clusters
      const rng = make_random_stream(42);
      for (let i = 0; i < n; i++) {
        const cluster = Math.floor(i / (n / 3));
        labels.push(cluster);

        // Add small noise to cluster centers
        const noise = () => (rng.rand() - 0.5) * 0.2;
        if (cluster === 0) {
          X.push([0 + noise(), 0 + noise()]);
        } else if (cluster === 1) {
          X.push([10 + noise(), 10 + noise()]);
        } else {
          X.push([10 + noise(), 0 + noise()]);
        }
      }
      
      const start = Date.now();
      const score = davies_bouldin_efficient(X, labels);
      const elapsed = Date.now() - start;
      
      expect(score).toBeLessThan(0.5); // Well-separated clusters
      expect(elapsed).toBeLessThan(1000); // Should complete in < 1 second
    });
  });

  describe("Comparison with sklearn", () => {
    it("should match sklearn for reference dataset", () => {
      const X = [
        [1, 1], [1.5, 2], [3, 4],
        [5, 7], [3.5, 5], [4.5, 5],
        [3.5, 4.5]
      ];
      const labels = [0, 0, 0, 1, 1, 1, 0];

      const score = davies_bouldin(X, labels);

      // sklearn.metrics.davies_bouldin_score on this input.
      expect(score).toBeCloseTo(0.7991550052810356, 6);
    });
  });
});
describe("davies_bouldin – noise (-1) awareness", () => {
  const two_clusters = [
    [0, 0],
    [0.1, 0],
    [5, 5],
    [5.1, 5],
  ];

  it("excludes noise so score equals the noise-free score", () => {
    const base = davies_bouldin(two_clusters, [0, 0, 1, 1]);
    const with_noise = davies_bouldin(
      [...two_clusters, [100, 100], [-100, -100]],
      [0, 0, 1, 1, -1, -1],
    );
    expect(with_noise).toBeCloseTo(base, 4);
  });

  it("returns defined 0 when every label is noise", () => {
    expect(davies_bouldin(two_clusters, [-1, -1, -1, -1])).toBe(0);
    expect(davies_bouldin_efficient(two_clusters, [-1, -1, -1, -1])).toBe(0);
  });

  it("returns defined 0 for a single cluster plus noise", () => {
    expect(davies_bouldin(two_clusters, [0, 0, 0, -1])).toBe(0);
    expect(davies_bouldin_efficient(two_clusters, [0, 0, 0, -1])).toBe(0);
  });

  it("supports the cosine metric", () => {
    const s = davies_bouldin(two_clusters, [0, 0, 1, 1], "cosine");
    expect(Number.isFinite(s)).toBe(true);
  });
});

describe("davies_bouldin – cosine metric centroid", () => {
  beforeEach(() => {
    tf.engine().startScope();
  });

  afterEach(() => {
    tf.engine().endScope();
  });

  // Under cosine, well-separated directional clusters score lower than
  // poorly-separated ones.
  it("gives lower score for well-separated cosine clusters than poorly-separated ones", () => {
    // Well-separated: cluster 0 points right (+x), cluster 1 points up (+y).
    const well_separated = [
      [1, 0.05], [0.95, 0.1],   // cluster 0: roughly +x direction
      [0.05, 1], [0.1, 0.95],   // cluster 1: roughly +y direction
    ];
    const labels_ws = [0, 0, 1, 1];

    // Poorly-separated: both clusters point diagonally, close in cosine space.
    const poorly_separated = [
      [1, 0.9], [0.9, 1],     // cluster 0: ~45 degrees
      [1, 1.1], [1.1, 0.9],   // cluster 1: ~45 degrees, adjacent
    ];
    const labels_ps = [0, 0, 1, 1];

    const score_ws = davies_bouldin(well_separated, labels_ws, "cosine");
    const score_ps = davies_bouldin(poorly_separated, labels_ps, "cosine");

    expect(score_ws).toBeLessThan(score_ps);
    expect(Number.isFinite(score_ws)).toBe(true);
    expect(Number.isFinite(score_ps)).toBe(true);
  });

  it("efficient version also gives lower score for well-separated cosine clusters", () => {
    const well_separated = [
      [1, 0.05], [0.95, 0.1],
      [0.05, 1], [0.1, 0.95],
    ];
    const poorly_separated = [
      [1, 0.9], [0.9, 1],
      [1, 1.1], [1.1, 0.9],
    ];

    const score_ws = davies_bouldin_efficient(well_separated, [0, 0, 1, 1], "cosine");
    const score_ps = davies_bouldin_efficient(poorly_separated, [0, 0, 1, 1], "cosine");

    expect(score_ws).toBeLessThan(score_ps);
    expect(Number.isFinite(score_ws)).toBe(true);
  });

  // When a cluster's Euclidean mean lies near the origin (near-antipodal
  // members), cluster_centroid normalises the mean to a unit vector so the
  // centroid is semantically valid in cosine space. Because cluster_dispersion
  // and pairwise_distance_matrix re-normalise internally, the DB score is
  // numerically invariant to this for any non-zero mean, so this test checks
  // the API contract: a finite positive score, consistent between both
  // functions. An exactly-antipodal cluster (mean = [0,…,0]) is undefined, so
  // each cluster carries a small directional bias to keep a clear residual
  // direction.
  it("produces a finite sensible score when clusters contain near-antipodal unit vectors", () => {
    const X = [
      // Cluster 0: nearly antipodal along x-axis, small +y bias → centroid direction [0, 1].
      [1, 0.05],   // near +x
      [-1, 0.05],  // near -x
      // Cluster 1: nearly antipodal along y-axis, small +x bias → centroid direction [1, 0].
      [0.05, 1],   // near +y
      [0.05, -1],  // near -y
    ];
    const labels = [0, 0, 1, 1];

    const score = davies_bouldin(X, labels, "cosine");
    const score_eff = davies_bouldin_efficient(X, labels, "cosine");

    // Must be a finite positive number – not NaN, not Infinity, not zero.
    expect(Number.isFinite(score)).toBe(true);
    expect(score).toBeGreaterThan(0);
    // Both functions must agree.
    expect(score_eff).toBeCloseTo(score, 5);
  });
});
