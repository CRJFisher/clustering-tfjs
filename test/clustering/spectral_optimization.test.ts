import * as tf from "../tensorflow-helper";
import { validationBasedOptimization } from "../../src/clustering/spectral_optimization";

describe("validationBasedOptimization", () => {
  const embeddingData = [
    [0, 0], [0.1, 0.1], [0.2, 0], [-0.1, 0.1], [0, -0.1],
    [5, 5], [5.1, 5.1], [5.2, 5], [4.9, 5.1], [5, 4.9],
  ];

  function makeEmbedding(): tf.Tensor2D {
    return tf.tensor2d(embeddingData);
  }

  it("returns valid labels with calinski-harabasz metric", async () => {
    const embedding = makeEmbedding();
    try {
      const result = await validationBasedOptimization(embedding, 2, "calinski-harabasz", 3, 42);
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
    const embedding = makeEmbedding();
    try {
      const result = await validationBasedOptimization(embedding, 2, "davies-bouldin", 3, 42);
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
    const embedding = makeEmbedding();
    try {
      const result = await validationBasedOptimization(embedding, 2, "silhouette", 3, 42);
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
    const embedding1 = makeEmbedding();
    const embedding2 = makeEmbedding();
    try {
      const result1 = await validationBasedOptimization(embedding1, 2, "calinski-harabasz", 5, 99);
      const result2 = await validationBasedOptimization(embedding2, 2, "calinski-harabasz", 5, 99);
      expect(result1.labels).toEqual(result2.labels);
    } finally {
      embedding1.dispose();
      embedding2.dispose();
    }
  });

  it("labels length matches embedding rows", async () => {
    const embedding = makeEmbedding();
    try {
      const result = await validationBasedOptimization(embedding, 2, "calinski-harabasz", 3, 42);
      expect(result.labels.length).toBe(embedding.shape[0]);
    } finally {
      embedding.dispose();
    }
  });

  it("all labels are valid cluster indices [0, nClusters)", async () => {
    const nClusters = 2;
    const embedding = makeEmbedding();
    try {
      const result = await validationBasedOptimization(embedding, nClusters, "silhouette", 3, 42);
      for (const l of result.labels) {
        expect(Number.isInteger(l)).toBe(true);
        expect(l).toBeGreaterThanOrEqual(0);
        expect(l).toBeLessThan(nClusters);
      }
    } finally {
      embedding.dispose();
    }
  });

  it("returns OptimizationResult with correct structure", async () => {
    const embedding = makeEmbedding();
    try {
      const result = await validationBasedOptimization(embedding, 2, "calinski-harabasz", 3, 42);
      expect(result).toHaveProperty("labels");
      expect(result).toHaveProperty("config");
      expect(Array.isArray(result.labels)).toBe(true);
      expect(result.config).toHaveProperty("gamma");
      expect(result.config).toHaveProperty("metric");
      expect(result.config).toHaveProperty("attempts");
      expect(result.config).toHaveProperty("useValidation");
    } finally {
      embedding.dispose();
    }
  });

  it("works with single attempt", async () => {
    const embedding = makeEmbedding();
    try {
      const result = await validationBasedOptimization(embedding, 2, "calinski-harabasz", 1, 42);
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
