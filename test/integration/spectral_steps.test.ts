import * as tf from "../tensorflow-helper";
import { SpectralClustering } from "../../src/clustering/spectral";
import * as fs from "fs";
import * as path from "path";

describe("SpectralClustering step-by-step verification", () => {
  const FIXTURE_DIR = path.join(__dirname, "../fixtures/spectral");
  
  // Helper to load fixture data
  function loadFixture(filename: string) {
    const filepath = path.join(FIXTURE_DIR, filename);
    return JSON.parse(fs.readFileSync(filepath, "utf8"));
  }

  // Helper to compare matrix statistics
  function getMatrixStats(tensor: tf.Tensor2D) {
    return tf.tidy(() => {
      const data = tensor.dataSync();
      const nnz = Array.from(data).filter(v => Math.abs(v) > 1e-10).length;
      const min = tensor.min().dataSync()[0];
      const max = tensor.max().dataSync()[0];
      const mean = tensor.mean().dataSync()[0];
      
      return { nnz, min, max, mean, shape: tensor.shape };
    });
  }

  describe("Affinity matrix computation", () => {
    test("RBF affinity matrix properties", async () => {
      const fixture = loadFixture("blobs_n2_rbf.json");
      const spectral = new SpectralClustering({
        nClusters: 2,
        affinity: "rbf",
        gamma: fixture.params.gamma,
        captureDebugInfo: true,
      });

      const X = tf.tensor2d(fixture.X);
      const result = await spectral.fitWithIntermediateSteps(X);
      const affinity = result.affinity;
      const stats = getMatrixStats(affinity);

      // RBF affinity should be:
      // - Square matrix
      expect(stats.shape[0]).toBe(stats.shape[1]);
      expect(stats.shape[0]).toBe(fixture.X.length);
      
      // - All values between 0 and 1
      expect(stats.min).toBeGreaterThanOrEqual(0);
      expect(stats.max).toBeLessThanOrEqual(1);
      
      // - Diagonal should be 1 (self-similarity)
      const affinityArray = await affinity.array() as number[][];
      for (let i = 0; i < affinityArray.length; i++) {
        expect(affinityArray[i][i]).toBeCloseTo(1, 5);
      }

      // - Symmetric
      const transpose = tf.transpose(affinity);
      const diff = tf.sub(affinity, transpose);
      const maxDiff = await tf.max(tf.abs(diff)).data();
      expect(maxDiff[0]).toBeLessThan(1e-10);

      X.dispose();
      affinity.dispose();
      transpose.dispose();
      diff.dispose();
      result.laplacian.laplacian.dispose();
      result.laplacian.degrees?.dispose();
      result.laplacian.sqrtDegrees?.dispose();
      result.embedding.embedding.dispose();
      result.embedding.eigenvalues.dispose();
      result.embedding.rawEigenvectors?.dispose();
    });

    test("k-NN affinity matrix properties", async () => {
      const fixture = loadFixture("blobs_n2_knn.json");
      const spectral = new SpectralClustering({
        nClusters: 2,
        affinity: "nearest_neighbors",
        nNeighbors: fixture.params.nNeighbors,
        captureDebugInfo: true,
      });

      const X = tf.tensor2d(fixture.X);
      const result = await spectral.fitWithIntermediateSteps(X);
      const affinity = result.affinity;
      const stats = getMatrixStats(affinity);

      // k-NN affinity should be:
      // - Square matrix
      expect(stats.shape[0]).toBe(stats.shape[1]);
      
      // - Sparse (many zeros)
      const sparsity = 1 - stats.nnz / (stats.shape[0] * stats.shape[1]);
      expect(sparsity).toBeGreaterThan(0.5); // At least 50% sparse
      
      // - Values 0, 0.5, or 1 (due to symmetrization)
      const data = await affinity.data();
      data.forEach(v => {
        expect(v === 0 || v === 0.5 || v === 1).toBeTruthy();
      });

      X.dispose();
      affinity.dispose();
      result.laplacian.laplacian.dispose();
      result.laplacian.degrees?.dispose();
      result.laplacian.sqrtDegrees?.dispose();
      result.embedding.embedding.dispose();
      result.embedding.eigenvalues.dispose();
      result.embedding.rawEigenvectors?.dispose();
    });
  });

  describe("Laplacian computation", () => {
    test("Normalized Laplacian properties", async () => {
      const fixture = loadFixture("circles_n2_rbf.json");
      const spectral = new SpectralClustering({
        nClusters: 2,
        affinity: "rbf",
        gamma: fixture.params.gamma,
        captureDebugInfo: true,
      });

      const X = tf.tensor2d(fixture.X);
      const result = await spectral.fitWithIntermediateSteps(X);
      const laplacian = result.laplacian.laplacian;

      // Normalized Laplacian should:
      // - Have diagonal close to 1
      const laplacianArray = await laplacian.array() as number[][];
      for (let i = 0; i < laplacianArray.length; i++) {
        expect(Math.abs(laplacianArray[i][i] - 1)).toBeLessThan(1e-10);
      }

      // - Be symmetric
      const transpose = tf.transpose(laplacian);
      const diff = tf.sub(laplacian, transpose);
      const maxDiff = await tf.max(tf.abs(diff)).data();
      expect(maxDiff[0]).toBeLessThan(1e-10);

      // Get debug info with spectrum
      const debugInfo = spectral.getDebugInfo();
      expect(debugInfo?.laplacianSpectrum).toBeDefined();
      
      // First eigenvalue should be close to 0 for connected graph
      const spectrum = debugInfo!.laplacianSpectrum!;
      expect(Math.abs(spectrum[0])).toBeLessThan(1e-7);

      X.dispose();
      result.affinity.dispose();
      laplacian.dispose();
      transpose.dispose();
      diff.dispose();
      result.laplacian.degrees?.dispose();
      result.laplacian.sqrtDegrees?.dispose();
      result.embedding.embedding.dispose();
      result.embedding.eigenvalues.dispose();
      result.embedding.rawEigenvectors?.dispose();
    });
  });

  describe("Spectral embedding", () => {
    test("Embedding dimensions and scaling", async () => {
      const fixture = loadFixture("moons_n2_knn.json");
      const spectral = new SpectralClustering({
        nClusters: 2,
        affinity: "nearest_neighbors",
        nNeighbors: fixture.params.nNeighbors,
        captureDebugInfo: true,
      });

      const X = tf.tensor2d(fixture.X);
      const result = await spectral.fitWithIntermediateSteps(X);
      const embedding = result.embedding.embedding;
      const eigenvalues = result.embedding.eigenvalues;

      // Embedding should have correct shape
      expect(embedding.shape).toEqual([fixture.X.length, 2]);

      // Check scaling factors
      const eigenData = await eigenvalues.data();
      eigenData.forEach(eigenval => {
        // Eigenvalues should be between 0 and 2 for normalized Laplacian
        expect(eigenval).toBeGreaterThanOrEqual(-0.5); // Allow some numerical error
        expect(eigenval).toBeLessThanOrEqual(2.1);
      });

      // Check debug info
      const debugInfo = spectral.getDebugInfo();
      expect(debugInfo?.embeddingStats).toBeDefined();
      expect(debugInfo!.embeddingStats!.uniqueValuesPerDim).toHaveLength(2);

      X.dispose();
      result.affinity.dispose();
      result.laplacian.laplacian.dispose();
      result.laplacian.degrees?.dispose();
      result.laplacian.sqrtDegrees?.dispose();
      embedding.dispose();
      eigenvalues.dispose();
      result.embedding.rawEigenvectors?.dispose();
    });
  });

  describe("Full pipeline with debug info", () => {
    test("Debug info captures all intermediate statistics", async () => {
      const fixture = loadFixture("blobs_n3_knn.json");
      const spectral = new SpectralClustering({
        nClusters: 3,
        affinity: "nearest_neighbors",
        nNeighbors: fixture.params.nNeighbors,
        captureDebugInfo: true,
        randomState: 42,
      });

      await spectral.fit(fixture.X);
      const labels = spectral.labels_;
      const debugInfo = spectral.getDebugInfo();

      // Should have captured all debug info
      expect(debugInfo).not.toBeNull();
      expect(debugInfo!.affinityStats).toBeDefined();
      expect(debugInfo!.laplacianSpectrum).toBeDefined();
      expect(debugInfo!.embeddingStats).toBeDefined();
      expect(debugInfo!.clusteringMetrics).toBeDefined();

      // Affinity stats
      expect(debugInfo!.affinityStats!.shape).toEqual([60, 60]);
      expect(debugInfo!.affinityStats!.nnz).toBeGreaterThan(0);

      // Laplacian spectrum (first few eigenvalues)
      expect(debugInfo!.laplacianSpectrum!.length).toBeGreaterThan(0);

      // Embedding stats
      expect(debugInfo!.embeddingStats!.shape).toEqual([60, 3]);
      expect(debugInfo!.embeddingStats!.uniqueValuesPerDim).toHaveLength(3);

      // Clustering metrics
      expect(debugInfo!.clusteringMetrics!.inertia).toBeGreaterThanOrEqual(0);
      expect(debugInfo!.clusteringMetrics!.iterations).toBeGreaterThanOrEqual(0);
      
      // Check labels
      expect(labels).toHaveLength(60);
      expect(new Set(labels).size).toBe(3);

      spectral.dispose();
    });
  });
});