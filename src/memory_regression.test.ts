import * as tf from '../test_support/tensorflow_helper';
import { KMeans } from './clustering/kmeans';
import { SpectralClustering } from './clustering/spectral';
import { SOM } from './clustering/som';
import { HDBSCAN } from './clustering/hdbscan';
import { find_optimal_clusters } from './model_selection/find_optimal_clusters';
import { export_for_visualization } from './visualization/som_visualization';
import { silhouette_score } from './validation/silhouette';
import { davies_bouldin, davies_bouldin_efficient } from './validation/davies_bouldin';
import { calinski_harabasz, calinski_harabasz_efficient } from './validation/calinski_harabasz';

/**
 * Memory regression tests that assert tensor count before/after fit+dispose
 * to ensure no tensor memory leaks in clustering algorithms.
 */
describe('Memory regression tests', () => {
  // Simple 2-cluster dataset
  const data: number[][] = [
    [0, 0], [0.1, 0.1], [0.2, 0.2],
    [10, 10], [10.1, 10.1], [9.9, 9.9],
  ];

  const labels = [0, 0, 0, 1, 1, 1];

  describe('KMeans', () => {
    it('fit() + dispose() should not leak tensors', async () => {
      const before = tf.memory().numTensors;

      const km = new KMeans({ n_clusters: 2, random_state: 42, n_init: 1 });
      await km.fit(data);
      km.dispose();

      const after = tf.memory().numTensors;
      expect(after).toBe(before);
    });

    it('fit() + dispose() with tensor input should not leak tensors', async () => {
      const X = tf.tensor2d(data);
      const before = tf.memory().numTensors;

      const km = new KMeans({ n_clusters: 2, random_state: 42, n_init: 1 });
      await km.fit(X);
      km.dispose();

      const after = tf.memory().numTensors;
      expect(after).toBe(before);

      X.dispose();
    });

    it('fit_predict() + dispose() should not leak tensors', async () => {
      const before = tf.memory().numTensors;

      const km = new KMeans({ n_clusters: 2, random_state: 42, n_init: 1 });
      await km.fit_predict(data);
      km.dispose();

      const after = tf.memory().numTensors;
      expect(after).toBe(before);
    });

    it('re-fitting should not leak tensors', async () => {
      const before = tf.memory().numTensors;

      const km = new KMeans({ n_clusters: 2, random_state: 42, n_init: 1 });
      await km.fit(data);
      // Re-fit: dispose() is called internally, freeing old centroids
      await km.fit(data);
      km.dispose();

      const after = tf.memory().numTensors;
      expect(after).toBe(before);
    });
  });

  describe('SpectralClustering', () => {
    it('fit() + dispose() with array input should not leak tensors', async () => {
      const before = tf.memory().numTensors;

      const sc = new SpectralClustering({ n_clusters: 2, random_state: 42 });
      await sc.fit(data);
      sc.dispose();

      const after = tf.memory().numTensors;
      expect(after).toBe(before);
    });

    it('fit() + dispose() with tensor input should not leak tensors', async () => {
      const X = tf.tensor2d(data);
      const before = tf.memory().numTensors;

      const sc = new SpectralClustering({ n_clusters: 2, random_state: 42 });
      await sc.fit(X);
      sc.dispose();

      const after = tf.memory().numTensors;
      // The input tensor X still exists, so after should equal before
      expect(after).toBe(before);

      X.dispose();
    });

    it('fitWithIntermediateSteps should not leak tensors after cleanup', async () => {
      const before = tf.memory().numTensors;

      const sc = new SpectralClustering({ n_clusters: 2, random_state: 42 });
      const result = await sc.fit_with_intermediate_steps(data);

      // Dispose all returned intermediate tensors
      result.affinity.dispose();
      result.laplacian.laplacian.dispose();
      result.laplacian.degrees?.dispose();
      result.laplacian.sqrt_degrees?.dispose();
      result.embedding.embedding.dispose();
      result.embedding.eigenvalues.dispose();
      result.embedding.raw_eigenvectors?.dispose();
      result.embedding.scaling_factors?.dispose();

      sc.dispose();

      const after = tf.memory().numTensors;
      expect(after).toBe(before);
    });
  });

  describe('SOM', () => {
    it('fit() + dispose() should not leak tensors', async () => {
      const before = tf.memory().numTensors;

      const som = new SOM({
        grid_width: 2,
        grid_height: 2,

        num_epochs: 3,
        random_state: 42,
      });
      await som.fit(data);
      som.dispose();

      const after = tf.memory().numTensors;
      // SOM internally may retain a small number of tensors from backend ops;
      // tolerance of 4 matches the existing SOM memory test in som.test.ts
      expect(after).toBeLessThanOrEqual(before + 4);
    });

    it('exportForVisualization should not leak tensors', async () => {
      const som = new SOM({
        grid_width: 2,
        grid_height: 2,

        num_epochs: 3,
        random_state: 42,
      });
      await som.fit(data);

      const before = tf.memory().numTensors;
      await export_for_visualization(som, 'json');
      await export_for_visualization(som, 'csv');
      const after = tf.memory().numTensors;

      expect(after).toBe(before);

      som.dispose();
    });
  });

  describe('findOptimalClusters', () => {
    it('should not leak tensors with kmeans algorithm', async () => {
      const before = tf.memory().numTensors;

      await find_optimal_clusters(data, {
        min_clusters: 2,
        max_clusters: 3,
        algorithm: 'kmeans',
        algorithm_params: { random_state: 42, n_init: 1 },
      });

      const after = tf.memory().numTensors;
      expect(after).toBe(before);
    });

    it('should not leak tensors with spectral algorithm', async () => {
      const before = tf.memory().numTensors;

      await find_optimal_clusters(data, {
        min_clusters: 2,
        max_clusters: 3,
        algorithm: 'spectral',
        algorithm_params: { random_state: 42 },
      });

      const after = tf.memory().numTensors;
      expect(after).toBe(before);
    });

    it('should not leak tensors with agglomerative algorithm', async () => {
      const before = tf.memory().numTensors;

      await find_optimal_clusters(data, {
        min_clusters: 2,
        max_clusters: 3,
        algorithm: 'agglomerative',
      });

      const after = tf.memory().numTensors;
      expect(after).toBe(before);
    });

    it('should not leak tensors with som algorithm', async () => {
      const before = tf.memory().numTensors;

      await find_optimal_clusters(data, {
        min_clusters: 2,
        max_clusters: 3,
        algorithm: 'som',
        algorithm_params: { num_epochs: 3, random_state: 42 },
      });

      const after = tf.memory().numTensors;
      // SOM retains a small, bounded number of backend tensors after dispose
      // (same tolerance as the standalone SOM memory tests above).
      expect(after).toBeLessThanOrEqual(before + 4);
    });
  });

  describe('HDBSCAN', () => {
    const hdbscan_params = { min_cluster_size: 2, min_samples: 2 };

    it('fit() + dispose() with array input should not leak tensors', async () => {
      const before = tf.memory().numTensors;

      const hdb = new HDBSCAN(hdbscan_params);
      await hdb.fit(data);
      hdb.dispose();

      const after = tf.memory().numTensors;
      expect(after).toBe(before);
    });

    it('fit() + dispose() with tensor input should not leak tensors', async () => {
      const X = tf.tensor2d(data);
      const before = tf.memory().numTensors;

      const hdb = new HDBSCAN(hdbscan_params);
      await hdb.fit(X);
      hdb.dispose();

      const after = tf.memory().numTensors;
      expect(after).toBe(before);

      X.dispose();
    });

    it('re-fitting should not leak tensors', async () => {
      const before = tf.memory().numTensors;

      const hdb = new HDBSCAN(hdbscan_params);
      await hdb.fit(data);
      await hdb.fit(data);
      hdb.dispose();

      const after = tf.memory().numTensors;
      expect(after).toBe(before);
    });

    it('n === 1 graceful noise path should not leak tensors', async () => {
      const single_point = [[1, 2]];
      const before = tf.memory().numTensors;

      const hdb = new HDBSCAN(hdbscan_params);
      await hdb.fit(single_point);
      hdb.dispose();

      const after = tf.memory().numTensors;
      expect(after).toBe(before);
    });

    it('precomputed metric should not leak tensors', async () => {
      const precomputed = [
        [0, 1, Math.sqrt(2)],
        [1, 0, 1],
        [Math.sqrt(2), 1, 0],
      ];
      const before = tf.memory().numTensors;

      const hdb = new HDBSCAN({ ...hdbscan_params, metric: 'precomputed' });
      await hdb.fit(precomputed);
      hdb.dispose();

      const after = tf.memory().numTensors;
      expect(after).toBe(before);
    });

    it(
      'large-n fit should not leak tensors',
      async () => {
        // n=500 is within the O(n²) memory ceiling; validates no argument-spread
        // RangeError arises from the JS tail at this scale.
        const n = 500;
        const large_data = Array.from({ length: n }, (_, i) => [
          Math.sin(i * 0.1),
          Math.cos(i * 0.1),
        ]);

        const before = tf.memory().numTensors;

        const hdb = new HDBSCAN({ min_cluster_size: 5 });
        await hdb.fit(large_data);
        hdb.dispose();

        const after = tf.memory().numTensors;
        expect(after).toBe(before);
      },
      30_000,
    );
  });

  describe('Validation functions', () => {
    it('silhouette_score should not leak tensors', () => {
      const before = tf.memory().numTensors;
      silhouette_score(data, labels);
      const after = tf.memory().numTensors;
      expect(after).toBe(before);
    });

    it('davies_bouldin should not leak tensors', () => {
      const before = tf.memory().numTensors;
      davies_bouldin(data, labels);
      const after = tf.memory().numTensors;
      expect(after).toBe(before);
    });

    it('davies_bouldin_efficient should not leak tensors', () => {
      const before = tf.memory().numTensors;
      davies_bouldin_efficient(data, labels);
      const after = tf.memory().numTensors;
      expect(after).toBe(before);
    });

    it('calinski_harabasz should not leak tensors', () => {
      const before = tf.memory().numTensors;
      calinski_harabasz(data, labels);
      const after = tf.memory().numTensors;
      expect(after).toBe(before);
    });

    it('calinski_harabasz_efficient should not leak tensors', () => {
      const before = tf.memory().numTensors;
      calinski_harabasz_efficient(data, labels);
      const after = tf.memory().numTensors;
      expect(after).toBe(before);
    });
  });
});
