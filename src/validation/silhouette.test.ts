import * as tf from "../../test_support/tensorflow_helper";
import { silhouette_score, silhouette_score_subset, silhouette_samples } from "./silhouette";
import { make_random_stream } from "../random";

describe("silhouette_score", () => {
  beforeEach(() => {
    tf.engine().startScope();
  });

  afterEach(() => {
    tf.engine().endScope();
  });

  describe("Basic functionality", () => {
    it("computes high score for well-separated 2D clusters", () => {
      const X = [
        // Cluster 0: centered around (0, 0)
        [0, 0], [0.1, 0.1], [0.2, 0], [-0.1, 0.1],
        // Cluster 1: centered around (10, 10)
        [10, 10], [10.1, 10.1], [9.9, 9.9], [10.1, 9.9]
      ];
      const labels = [0, 0, 0, 0, 1, 1, 1, 1];

      const score = silhouette_score(X, labels);

      expect(score).toBeGreaterThan(0.8);
      expect(score).toBeLessThanOrEqual(1.0);
    });

    it("computes low score for overlapping clusters", () => {
      const X = [
        [0, 0], [0.5, 0.5], [1, 1], [1.5, 1.5],
        [0.5, 0], [1, 0.5], [1.5, 1], [2, 1.5]
      ];
      const labels = [0, 0, 0, 0, 1, 1, 1, 1];

      const score = silhouette_score(X, labels);

      expect(score).toBeLessThan(0.5);
      expect(score).toBeGreaterThan(-1.0);
    });

    it("computes negative score for misclassified points", () => {
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

      const score = silhouette_score(X, labels);

      expect(score).toBeLessThan(0.5);
    });

    it("computes same score with tensor inputs", () => {
      const X = [[1, 2], [1.5, 1.8], [5, 8], [8, 8], [1, 0.6], [9, 11]];
      const labels = [0, 0, 1, 1, 0, 1];

      const score_array = silhouette_score(X, labels);

      const X_tensor = tf.tensor2d(X);
      const labels_tensor = tf.tensor1d(labels);
      const score_tensor = silhouette_score(X_tensor, labels_tensor);

      expect(score_tensor).toBeCloseTo(score_array, 5);

      X_tensor.dispose();
      labels_tensor.dispose();
    });
  });

  describe("Edge cases", () => {
    it("throws for single cluster", () => {
      const X = [[1, 2], [3, 4], [5, 6]];
      const labels = [0, 0, 0];

      expect(() => silhouette_score(X, labels)).toThrow(
        "Silhouette score requires at least 2 clusters"
      );
    });

    it("handles single-point clusters", () => {
      const X = [
        [0, 0], [0.1, 0.1], // Cluster 0 (2 points)
        [5, 5],             // Cluster 1 (1 point)
        [10, 10], [10.1, 10.1] // Cluster 2 (2 points)
      ];
      const labels = [0, 0, 1, 2, 2];

      const score = silhouette_score(X, labels);
      expect(isFinite(score)).toBe(true);
      expect(score).toBeGreaterThanOrEqual(-1);
      expect(score).toBeLessThanOrEqual(1);
    });

    it("handles clusters of different sizes", () => {
      const X = [
        // Large cluster 0
        [0, 0], [0.1, 0.1], [0.2, 0], [-0.1, 0.1], [0, -0.1],
        // Small cluster 1
        [5, 5], [5.1, 5.1],
        // Medium cluster 2
        [0, 5], [0.1, 5], [-0.1, 5]
      ];
      const labels = [0, 0, 0, 0, 0, 1, 1, 2, 2, 2];

      const score = silhouette_score(X, labels);
      expect(isFinite(score)).toBe(true);
      expect(score).toBeGreaterThan(0);
    });
  });

  describe("Known values validation", () => {
    it("computes perfect score for perfectly separated clusters", () => {
      // Two clusters far apart
      const X = [
        [0, 0], [0, 0], [0, 0],    // Cluster 0 (all same point)
        [100, 100], [100, 100], [100, 100]  // Cluster 1 (all same point, far away)
      ];
      const labels = [0, 0, 0, 1, 1, 1];

      const score = silhouette_score(X, labels);

      expect(score).toBeCloseTo(1.0, 2);
    });

    it("computes zero score for points on decision boundary", () => {
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

      const score = silhouette_score(X, labels);

      expect(Math.abs(score)).toBeLessThan(0.2);
    });
  });

  describe("Subset computation", () => {
    it("computes same score for subset as full computation", () => {
      const X = [
        [0, 0], [0.1, 0.1], [0.2, 0], [-0.1, 0.1],
        [5, 5], [5.1, 5.1], [4.9, 4.9], [5.1, 4.9]
      ];
      const labels = [0, 0, 0, 0, 1, 1, 1, 1];

      const full_score = silhouette_score(X, labels);

      const all_indices = Array.from({ length: 8 }, (_, i) => i);
      const subset_score = silhouette_score_subset(X, labels, all_indices);

      expect(subset_score).toBeCloseTo(full_score, 5);
    });

    it("handles partial subset correctly", () => {
      const X = [
        [0, 0], [0.1, 0.1], [0.2, 0], [-0.1, 0.1],
        [5, 5], [5.1, 5.1], [4.9, 4.9], [5.1, 4.9]
      ];
      const labels = [0, 0, 0, 0, 1, 1, 1, 1];

      const subset_indices = [0, 2, 4, 6]; // Sample from each cluster
      const score = silhouette_score_subset(X, labels, subset_indices);

      expect(score).toBeGreaterThan(0.5);
      expect(score).toBeLessThanOrEqual(1.0);
    });
  });

  describe("Numerical stability", () => {
    it("handles identical points in same cluster", () => {
      const X = [
        [1, 1], [1, 1], [1, 1],    // Identical points in cluster 0
        [5, 5], [5.1, 5.1], [4.9, 4.9]  // Cluster 1
      ];
      const labels = [0, 0, 0, 1, 1, 1];

      const score = silhouette_score(X, labels);
      expect(isFinite(score)).toBe(true);
      expect(score).toBeGreaterThan(0);
    });

    it("handles very small distances", () => {
      const X = [
        [0, 0], [0.0001, 0.0001], [0.0002, 0],
        [1, 1], [1.0001, 1.0001], [1.0002, 1]
      ];
      const labels = [0, 0, 0, 1, 1, 1];

      const score = silhouette_score(X, labels);
      expect(isFinite(score)).toBe(true);
      expect(score).toBeGreaterThan(0);
    });
  });

  describe("Performance", () => {
    it("subset computation produces similar score to full with 10% sample", () => {
      const n = 500;
      const X: number[][] = [];
      const labels: number[] = [];

      const rng = make_random_stream(42);
      for (let i = 0; i < n; i++) {
        const cluster = i < n/2 ? 0 : 1;
        labels.push(cluster);

        const noise = () => (rng.rand() - 0.5) * 0.5;
        if (cluster === 0) {
          X.push([0 + noise(), 0 + noise()]);
        } else {
          X.push([5 + noise(), 5 + noise()]);
        }
      }

      const full_score = silhouette_score(X, labels);

      const subset_size = Math.floor(n * 0.1);
      const subset_indices = Array.from({ length: subset_size },
        (_, i) => Math.floor(i * n / subset_size));

      const subset_score = silhouette_score_subset(X, labels, subset_indices);

      // On small datasets overhead dominates; just verify scores agree
      expect(Math.abs(full_score - subset_score)).toBeLessThan(0.1);
    });
  });

  describe("Zero-distance (identical points) handling", () => {
    it("returns 0 when a==0 and b==0 (all identical points)", () => {
      const X = [
        [1, 1], [1, 1], [1, 1],
        [1, 1], [1, 1], [1, 1],
      ];
      const labels = [0, 0, 0, 1, 1, 1];

      const score = silhouette_score(X, labels);
      expect(score).toBe(0);
      expect(isNaN(score)).toBe(false);
    });

    it("returns 0 for identical points using silhouette_score_subset", () => {
      const X = [
        [1, 1], [1, 1], [1, 1],
        [1, 1], [1, 1], [1, 1],
      ];
      const labels = [0, 0, 0, 1, 1, 1];
      const all_indices = [0, 1, 2, 3, 4, 5];

      const score = silhouette_score_subset(X, labels, all_indices);
      expect(score).toBe(0);
      expect(isNaN(score)).toBe(false);
    });

    it("handles mixed: some identical, some not", () => {
      const X = [
        [0, 0], [0, 0], [0, 0],
        [5, 5], [5.1, 5.1], [4.9, 4.9],
      ];
      const labels = [0, 0, 0, 1, 1, 1];

      const score = silhouette_score(X, labels);
      expect(isFinite(score)).toBe(true);
      expect(isNaN(score)).toBe(false);
      expect(score).toBeGreaterThan(0);
    });
  });

  describe("Labels length validation", () => {
    it("throws when labels length mismatches data rows (array)", () => {
      const X = [[1, 2], [3, 4], [5, 6]];
      const labels = [0, 1];

      expect(() => silhouette_score(X, labels)).toThrow(
        "Labels length (2) does not match data rows (3)"
      );
    });

    it("throws when labels length mismatches data rows (tensor)", () => {
      const X = tf.tensor2d([[1, 2], [3, 4], [5, 6]]);
      const labels = tf.tensor1d([0, 1]);

      expect(() => silhouette_score(X, labels)).toThrow(
        "Labels length (2) does not match data rows (3)"
      );

      X.dispose();
      labels.dispose();
    });

    it("throws for silhouette_score_subset with mismatched labels", () => {
      const X = [[1, 2], [3, 4], [5, 6]];
      const labels = [0, 1];

      expect(() => silhouette_score_subset(X, labels, [0])).toThrow(
        "Labels length (2) does not match data rows (3)"
      );
    });
  });

  describe("silhouette_samples", () => {
    it("returns correct number of per-sample scores", () => {
      const X = [
        [0, 0], [0.1, 0.1], [0.2, 0],
        [5, 5], [5.1, 5.1], [4.9, 4.9]
      ];
      const labels = [0, 0, 0, 1, 1, 1];

      const samples = silhouette_samples(X, labels);
      expect(samples).toHaveLength(6);
    });

    it("mean equals silhouette_score", () => {
      const X = [
        [0, 0], [0.1, 0.1], [0.2, 0], [-0.1, 0.1],
        [5, 5], [5.1, 5.1], [4.9, 4.9], [5.1, 4.9]
      ];
      const labels = [0, 0, 0, 0, 1, 1, 1, 1];

      const samples = silhouette_samples(X, labels);
      const mean = samples.reduce((s, v) => s + v, 0) / samples.length;
      const score = silhouette_score(X, labels);

      expect(mean).toBeCloseTo(score, 10);
    });

    it("matches hand-computed per-sample coefficients", () => {
      // Collinear points: cluster 0 = {[0,0],[2,0]}, cluster 1 = {[10,0],[12,0]}.
      // sample [0,0]:  a=2, b=mean(10,12)=11 -> (11-2)/11 = 9/11
      // sample [2,0]:  a=2, b=mean(8,10)=9   -> (9-2)/9  = 7/9
      // sample [10,0]: a=2, b=mean(10,8)=9   -> (9-2)/9  = 7/9
      // sample [12,0]: a=2, b=mean(12,10)=11 -> (11-2)/11 = 9/11
      const X = [
        [0, 0],
        [2, 0],
        [10, 0],
        [12, 0],
      ];
      const labels = [0, 0, 1, 1];
      const samples = silhouette_samples(X, labels);
      expect(samples[0]).toBeCloseTo(9 / 11, 5);
      expect(samples[1]).toBeCloseTo(7 / 9, 5);
      expect(samples[2]).toBeCloseTo(7 / 9, 5);
      expect(samples[3]).toBeCloseTo(9 / 11, 5);
    });

    it("all values are in [-1, 1]", () => {
      const X = [
        [0, 0], [0.1, 0.1], [5, 5], [5.1, 5.1]
      ];
      const labels = [0, 0, 1, 1];

      const samples = silhouette_samples(X, labels);
      for (const s of samples) {
        expect(s).toBeGreaterThanOrEqual(-1);
        expect(s).toBeLessThanOrEqual(1);
      }
    });

    it("works with tensor inputs", () => {
      const X = [[1, 2], [1.5, 1.8], [5, 8], [8, 8], [1, 0.6], [9, 11]];
      const labels = [0, 0, 1, 1, 0, 1];

      const samples_array = silhouette_samples(X, labels);

      const X_tensor = tf.tensor2d(X);
      const labels_tensor = tf.tensor1d(labels);
      const samples_tensor = silhouette_samples(X_tensor, labels_tensor);

      for (let i = 0; i < samples_array.length; i++) {
        expect(samples_tensor[i]).toBeCloseTo(samples_array[i], 5);
      }

      X_tensor.dispose();
      labels_tensor.dispose();
    });

    it("throws on single cluster", () => {
      const X = [[1, 2], [3, 4]];
      const labels = [0, 0];

      expect(() => silhouette_samples(X, labels)).toThrow(
        "Silhouette score requires at least 2 clusters"
      );
    });

    it("throws when labels length mismatches data rows", () => {
      const X = [[1, 2], [3, 4], [5, 6]];
      const labels = [0, 1];

      expect(() => silhouette_samples(X, labels)).toThrow(
        "Labels length (2) does not match data rows (3)"
      );
    });
  });
});
describe("silhouette_score – noise (-1) awareness", () => {
  const two_clusters = [
    [0, 0],
    [0.1, 0],
    [5, 5],
    [5.1, 5],
  ];

  it("excludes noise so score equals the noise-free score", () => {
    const base = silhouette_score(two_clusters, [0, 0, 1, 1]);
    // Same four cluster points plus two noise points that must be ignored.
    const with_noise = silhouette_score(
      [...two_clusters, [100, 100], [-100, -100]],
      [0, 0, 1, 1, -1, -1],
    );
    expect(with_noise).toBeCloseTo(base, 6);
  });

  it("throws when every label is noise", () => {
    expect(() => silhouette_score(two_clusters, [-1, -1, -1, -1])).toThrow(
      "all labels are noise",
    );
    expect(() => silhouette_samples(two_clusters, [-1, -1, -1, -1])).toThrow(
      "all labels are noise",
    );
  });

  it("returns defined 0 for a single cluster plus noise", () => {
    expect(silhouette_score(two_clusters, [0, 0, 0, -1])).toBe(0);
  });

  it("silhouette_score_subset ignores noise samples and neighbours", () => {
    const base = silhouette_score_subset(two_clusters, [0, 0, 1, 1], [0, 2]);
    const with_noise = silhouette_score_subset(
      [...two_clusters, [100, 100]],
      [0, 0, 1, 1, -1],
      [0, 2, 4],
    );
    expect(with_noise).toBeCloseTo(base, 6);
    // All-noise dataset throws.
    expect(() =>
      silhouette_score_subset(two_clusters, [-1, -1, -1, -1], [0, 1]),
    ).toThrow("all labels are noise");
  });

  it("supports the cosine metric", () => {
    const s = silhouette_score(two_clusters, [0, 0, 1, 1], "cosine");
    expect(Number.isFinite(s)).toBe(true);
  });
});

describe("silhouette_score – degenerate-input contract", () => {
  const X2 = [[0, 0], [1, 0]];

  it("all-noise labels throw with a descriptive message", () => {
    const msg = "all labels are noise";
    expect(() => silhouette_score(X2, [-1, -1])).toThrow(msg);
    expect(() => silhouette_samples(X2, [-1, -1])).toThrow(msg);
    expect(() => silhouette_score_subset(X2, [-1, -1], [0])).toThrow(msg);
  });

  it("single-cluster labels (no noise) throw", () => {
    const msg = "Silhouette score requires at least 2 clusters";
    expect(() => silhouette_score(X2, [0, 0])).toThrow(msg);
    expect(() => silhouette_samples(X2, [0, 0])).toThrow(msg);
    expect(() => silhouette_score_subset(X2, [0, 0], [0])).toThrow(msg);
  });

  it("two-point two-cluster input returns a finite score", () => {
    const score = silhouette_score(X2, [0, 1]);
    expect(Number.isFinite(score)).toBe(true);
    expect(score).toBeGreaterThanOrEqual(-1);
    expect(score).toBeLessThanOrEqual(1);
  });

  it("n_samples=0 throws on labels length mismatch", () => {
    expect(() => silhouette_score([], [0])).toThrow();
  });

  it("single cluster plus noise returns defined 0, not a throw", () => {
    expect(silhouette_score([[0, 0], [1, 0], [2, 0]], [0, 0, -1])).toBe(0);
  });
});
