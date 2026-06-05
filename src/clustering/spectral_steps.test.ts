import * as tf from "../../test_support/tensorflow_helper";
import { SpectralClustering } from "./spectral";
import * as fs from "fs";
import * as path from "path";

describe("SpectralClustering step-by-step verification", () => {
  // Use path relative to project root for fixtures
  const FIXTURE_DIR = path.join(process.cwd(), "__fixtures__", "spectral");
  
  // Helper to load fixture data
  function load_fixture(filename: string) {
    const filepath = path.join(FIXTURE_DIR, filename);
    return JSON.parse(fs.readFileSync(filepath, "utf8"));
  }

  // Helper to compare matrix statistics
  function get_matrix_stats(tensor: tf.Tensor2D) {
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
      const fixture = load_fixture("blobs_n2_rbf.json");
      const spectral = new SpectralClustering({
        n_clusters: 2,
        affinity: "rbf",
        gamma: fixture.params.gamma,
        capture_debug_info: true,
      });

      const X = tf.tensor2d(fixture.X);
      const result = await spectral.fit_with_intermediate_steps(X);
      const affinity = result.affinity;
      const stats = get_matrix_stats(affinity);

      // RBF affinity should be:
      // - Square matrix
      expect(stats.shape[0]).toBe(stats.shape[1]);
      expect(stats.shape[0]).toBe(fixture.X.length);
      
      // - All values between 0 and 1
      expect(stats.min).toBeGreaterThanOrEqual(0);
      expect(stats.max).toBeLessThanOrEqual(1);
      
      // - Diagonal should be 1 (self-similarity)
      const affinity_array = await affinity.array() as number[][];
      for (let i = 0; i < affinity_array.length; i++) {
        expect(affinity_array[i][i]).toBeCloseTo(1, 5);
      }

      // - Symmetric
      const transpose = tf.transpose(affinity);
      const diff = tf.sub(affinity, transpose);
      const max_diff = await tf.max(tf.abs(diff)).data();
      expect(max_diff[0]).toBeLessThan(1e-10);

      X.dispose();
      affinity.dispose();
      transpose.dispose();
      diff.dispose();
      result.laplacian.laplacian.dispose();
      result.laplacian.degrees?.dispose();
      result.laplacian.sqrt_degrees?.dispose();
      result.embedding.embedding.dispose();
      result.embedding.eigenvalues.dispose();
      result.embedding.raw_eigenvectors?.dispose();
    });

    test("k-NN affinity matrix properties", async () => {
      const fixture = load_fixture("blobs_n2_knn.json");
      const spectral = new SpectralClustering({
        n_clusters: 2,
        affinity: "nearest_neighbors",
        n_neighbors: fixture.params.n_neighbors,
        capture_debug_info: true,
      });

      const X = tf.tensor2d(fixture.X);
      const result = await spectral.fit_with_intermediate_steps(X);
      const affinity = result.affinity;
      const stats = get_matrix_stats(affinity);

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
      result.laplacian.sqrt_degrees?.dispose();
      result.embedding.embedding.dispose();
      result.embedding.eigenvalues.dispose();
      result.embedding.raw_eigenvectors?.dispose();
    });
  });

  describe("Laplacian computation", () => {
    test("Normalized Laplacian properties", async () => {
      const fixture = load_fixture("circles_n2_rbf.json");
      const spectral = new SpectralClustering({
        n_clusters: 2,
        affinity: "rbf",
        gamma: fixture.params.gamma,
        capture_debug_info: true,
      });

      const X = tf.tensor2d(fixture.X);
      const result = await spectral.fit_with_intermediate_steps(X);
      const laplacian = result.laplacian.laplacian;

      // Normalized Laplacian should:
      // - Have diagonal close to 1
      const laplacian_array = await laplacian.array() as number[][];
      for (let i = 0; i < laplacian_array.length; i++) {
        expect(Math.abs(laplacian_array[i][i] - 1)).toBeLessThan(1e-10);
      }

      // - Be symmetric
      const transpose = tf.transpose(laplacian);
      const diff = tf.sub(laplacian, transpose);
      const max_diff = await tf.max(tf.abs(diff)).data();
      expect(max_diff[0]).toBeLessThan(1e-10);

      // Get debug info with spectrum
      const debug_info = spectral.get_debug_info();
      expect(debug_info?.laplacian_spectrum).toBeDefined();
      
      // First eigenvalue should be close to 0 for connected graph
      const spectrum = debug_info!.laplacian_spectrum!;
      expect(Math.abs(spectrum[0])).toBeLessThan(1e-7);

      X.dispose();
      result.affinity.dispose();
      laplacian.dispose();
      transpose.dispose();
      diff.dispose();
      result.laplacian.degrees?.dispose();
      result.laplacian.sqrt_degrees?.dispose();
      result.embedding.embedding.dispose();
      result.embedding.eigenvalues.dispose();
      result.embedding.raw_eigenvectors?.dispose();
    });
  });

  describe("Spectral embedding", () => {
    test("Embedding dimensions and scaling", async () => {
      const fixture = load_fixture("moons_n2_knn.json");
      const spectral = new SpectralClustering({
        n_clusters: 2,
        affinity: "nearest_neighbors",
        n_neighbors: fixture.params.n_neighbors,
        capture_debug_info: true,
      });

      const X = tf.tensor2d(fixture.X);
      const result = await spectral.fit_with_intermediate_steps(X);
      const embedding = result.embedding.embedding;
      const eigenvalues = result.embedding.eigenvalues;

      // Embedding should have correct shape
      expect(embedding.shape).toEqual([fixture.X.length, 2]);

      // Check scaling factors
      const eigen_data = await eigenvalues.data();
      eigen_data.forEach(eigenval => {
        // Eigenvalues should be between 0 and 2 for normalized Laplacian
        expect(eigenval).toBeGreaterThanOrEqual(-0.5); // Allow some numerical error
        expect(eigenval).toBeLessThanOrEqual(2.1);
      });

      // Check debug info
      const debug_info = spectral.get_debug_info();
      expect(debug_info?.embedding_stats).toBeDefined();
      expect(debug_info!.embedding_stats!.unique_values_per_dim).toHaveLength(2);

      X.dispose();
      result.affinity.dispose();
      result.laplacian.laplacian.dispose();
      result.laplacian.degrees?.dispose();
      result.laplacian.sqrt_degrees?.dispose();
      embedding.dispose();
      eigenvalues.dispose();
      result.embedding.raw_eigenvectors?.dispose();
    });
  });

  describe("Full pipeline with debug info", () => {
    test("Debug info captures all intermediate statistics", async () => {
      const fixture = load_fixture("blobs_n3_knn.json");
      const spectral = new SpectralClustering({
        n_clusters: 3,
        affinity: "nearest_neighbors",
        n_neighbors: fixture.params.n_neighbors,
        capture_debug_info: true,
        random_state: 42,
      });

      await spectral.fit(fixture.X);
      const labels = spectral.labels_;
      const debug_info = spectral.get_debug_info();

      // Should have captured all debug info
      expect(debug_info).not.toBeNull();
      expect(debug_info!.affinity_stats).toBeDefined();
      expect(debug_info!.laplacian_spectrum).toBeDefined();
      expect(debug_info!.embedding_stats).toBeDefined();
      expect(debug_info!.clustering_metrics).toBeDefined();

      // Affinity stats
      expect(debug_info!.affinity_stats!.shape).toEqual([60, 60]);
      expect(debug_info!.affinity_stats!.nnz).toBeGreaterThan(0);

      // Laplacian spectrum (first few eigenvalues)
      expect(debug_info!.laplacian_spectrum!.length).toBeGreaterThan(0);

      // Embedding stats
      expect(debug_info!.embedding_stats!.shape).toEqual([60, 3]);
      expect(debug_info!.embedding_stats!.unique_values_per_dim).toHaveLength(3);

      // Clustering metrics
      expect(debug_info!.clustering_metrics!.inertia).toBeGreaterThanOrEqual(0);
      expect(debug_info!.clustering_metrics!.iterations).toBeGreaterThanOrEqual(0);
      
      // Check labels
      expect(labels).toHaveLength(60);
      expect(new Set(labels).size).toBe(3);

      spectral.dispose();
    });
  });
});