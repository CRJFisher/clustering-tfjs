import * as tf from "../../test_support/tensorflow_helper";
import { validation_based_optimization, intensive_parameter_sweep } from "./spectral_optimization";
import type { SpectralClusteringParams } from "./types";

describe("validationBasedOptimization", () => {
  const embedding_data = [
    [0, 0], [0.1, 0.1], [0.2, 0], [-0.1, 0.1], [0, -0.1],
    [5, 5], [5.1, 5.1], [5.2, 5], [4.9, 5.1], [5, 4.9],
  ];

  function make_embedding(): tf.Tensor2D {
    return tf.tensor2d(embedding_data);
  }

  it("returns valid labels with calinski-harabasz metric", async () => {
    const embedding = make_embedding();
    try {
      const result = await validation_based_optimization(embedding, 2, "calinski-harabasz", 3, 42);
      expect(result.labels.length).toBe(10);
      for (const l of result.labels) {
        expect(l).toBeGreaterThanOrEqual(0);
        expect(l).toBeLessThan(2);
      }
      expect(new Set(result.labels).size).toBe(2);
    } finally {
      embedding.dispose();
    }
  });

  it("returns valid labels with davies-bouldin metric", async () => {
    const embedding = make_embedding();
    try {
      const result = await validation_based_optimization(embedding, 2, "davies-bouldin", 3, 42);
      expect(result.labels.length).toBe(10);
      for (const l of result.labels) {
        expect(l).toBeGreaterThanOrEqual(0);
        expect(l).toBeLessThan(2);
      }
      expect(new Set(result.labels).size).toBe(2);
    } finally {
      embedding.dispose();
    }
  });

  it("returns valid labels with silhouette metric", async () => {
    const embedding = make_embedding();
    try {
      const result = await validation_based_optimization(embedding, 2, "silhouette", 3, 42);
      expect(result.labels.length).toBe(10);
      for (const l of result.labels) {
        expect(l).toBeGreaterThanOrEqual(0);
        expect(l).toBeLessThan(2);
      }
      expect(new Set(result.labels).size).toBe(2);
    } finally {
      embedding.dispose();
    }
  });

  it("is deterministic with same randomState", async () => {
    const embedding1 = make_embedding();
    const embedding2 = make_embedding();
    try {
      const result1 = await validation_based_optimization(embedding1, 2, "calinski-harabasz", 5, 99);
      const result2 = await validation_based_optimization(embedding2, 2, "calinski-harabasz", 5, 99);
      expect(result1.labels).toEqual(result2.labels);
    } finally {
      embedding1.dispose();
      embedding2.dispose();
    }
  });

  it("labels length matches embedding rows", async () => {
    const embedding = make_embedding();
    try {
      const result = await validation_based_optimization(embedding, 2, "calinski-harabasz", 3, 42);
      expect(result.labels.length).toBe(embedding.shape[0]);
    } finally {
      embedding.dispose();
    }
  });

  it("all labels are valid cluster indices [0, n_clusters)", async () => {
    const n_clusters = 2;
    const embedding = make_embedding();
    try {
      const result = await validation_based_optimization(embedding, n_clusters, "silhouette", 3, 42);
      for (const l of result.labels) {
        expect(Number.isInteger(l)).toBe(true);
        expect(l).toBeGreaterThanOrEqual(0);
        expect(l).toBeLessThan(n_clusters);
      }
    } finally {
      embedding.dispose();
    }
  });

  it("returns OptimizationResult with correct structure", async () => {
    const embedding = make_embedding();
    try {
      const result = await validation_based_optimization(embedding, 2, "calinski-harabasz", 3, 42);
      expect(result).toHaveProperty("labels");
      expect(result).toHaveProperty("config");
      expect(Array.isArray(result.labels)).toBe(true);
      expect(result.config).toHaveProperty("gamma");
      expect(result.config).toHaveProperty("metric");
      expect(result.config).toHaveProperty("attempts");
      expect(result.config).toHaveProperty("use_validation");
    } finally {
      embedding.dispose();
    }
  });

  it("works with single attempt", async () => {
    const embedding = make_embedding();
    try {
      const result = await validation_based_optimization(embedding, 2, "calinski-harabasz", 1, 42);
      expect(result.labels.length).toBe(10);
      for (const l of result.labels) {
        expect(l).toBeGreaterThanOrEqual(0);
        expect(l).toBeLessThan(2);
      }
    } finally {
      embedding.dispose();
    }
  });
});

describe("intensive_parameter_sweep", () => {
  it("throws when all gamma attempts produce degenerate embeddings", async () => {
    const n = 6;
    const X = tf.zeros([n, 2]) as tf.Tensor2D;
    const params: SpectralClusteringParams = {
      n_clusters: 2,
      affinity: 'rbf',
      gamma_range: [1.0],
    };

    // All-zero embedding: metric scores become NaN (or the metric throws for k≤1),
    // so no gamma ever beats the -Infinity baseline.
    const degenerate_embedding = async (a: tf.Tensor2D): Promise<tf.Tensor2D> => {
      a.dispose();
      return tf.zeros([n, 2]) as tf.Tensor2D;
    };

    const trivial_affinity = (_x: tf.Tensor2D, _p: SpectralClusteringParams): tf.Tensor2D => {
      return tf.eye(n) as tf.Tensor2D;
    };

    try {
      await expect(
        intensive_parameter_sweep(X, params, degenerate_embedding, trivial_affinity),
      ).rejects.toThrow(/all gamma attempts produced degenerate embeddings/);
    } finally {
      X.dispose();
    }
  });
});
