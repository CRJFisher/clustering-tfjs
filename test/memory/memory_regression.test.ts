import * as tf from '../tensorflow-helper';
import { KMeans } from '../../src/clustering/kmeans';
import { SpectralClustering } from '../../src/clustering/spectral';
import { SOM } from '../../src/clustering/som';
import { findOptimalClusters } from '../../src/utils/findOptimalClusters';
import { exportForVisualization } from '../../src/utils/som_visualization';
import { silhouetteScore } from '../../src/validation/silhouette';
import { daviesBouldin, daviesBouldinEfficient } from '../../src/validation/davies_bouldin';
import { calinskiHarabasz, calinskiHarabaszEfficient } from '../../src/validation/calinski_harabasz';

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

      const km = new KMeans({ nClusters: 2, randomState: 42, nInit: 1 });
      await km.fit(data);
      km.dispose();

      const after = tf.memory().numTensors;
      expect(after).toBe(before);
    });

    it('fit() + dispose() with tensor input should not leak tensors', async () => {
      const X = tf.tensor2d(data);
      const before = tf.memory().numTensors;

      const km = new KMeans({ nClusters: 2, randomState: 42, nInit: 1 });
      await km.fit(X);
      km.dispose();

      const after = tf.memory().numTensors;
      expect(after).toBe(before);

      X.dispose();
    });

    it('fitPredict() + dispose() should not leak tensors', async () => {
      const before = tf.memory().numTensors;

      const km = new KMeans({ nClusters: 2, randomState: 42, nInit: 1 });
      await km.fitPredict(data);
      km.dispose();

      const after = tf.memory().numTensors;
      expect(after).toBe(before);
    });

    it('re-fitting should not leak tensors', async () => {
      const before = tf.memory().numTensors;

      const km = new KMeans({ nClusters: 2, randomState: 42, nInit: 1 });
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

      const sc = new SpectralClustering({ nClusters: 2, randomState: 42 });
      await sc.fit(data);
      sc.dispose();

      const after = tf.memory().numTensors;
      expect(after).toBe(before);
    });

    it('fit() + dispose() with tensor input should not leak tensors', async () => {
      const X = tf.tensor2d(data);
      const before = tf.memory().numTensors;

      const sc = new SpectralClustering({ nClusters: 2, randomState: 42 });
      await sc.fit(X);
      sc.dispose();

      const after = tf.memory().numTensors;
      // The input tensor X still exists, so after should equal before
      expect(after).toBe(before);

      X.dispose();
    });

    it('fitWithIntermediateSteps should not leak tensors after cleanup', async () => {
      const before = tf.memory().numTensors;

      const sc = new SpectralClustering({ nClusters: 2, randomState: 42 });
      const result = await sc.fitWithIntermediateSteps(data);

      // Dispose all returned intermediate tensors
      result.affinity.dispose();
      result.laplacian.laplacian.dispose();
      result.laplacian.degrees?.dispose();
      result.laplacian.sqrtDegrees?.dispose();
      result.embedding.embedding.dispose();
      result.embedding.eigenvalues.dispose();
      result.embedding.rawEigenvectors?.dispose();
      result.embedding.scalingFactors?.dispose();

      sc.dispose();

      const after = tf.memory().numTensors;
      expect(after).toBe(before);
    });
  });

  describe('SOM', () => {
    it('fit() + dispose() should not leak tensors', async () => {
      const before = tf.memory().numTensors;

      const som = new SOM({
        gridWidth: 2,
        gridHeight: 2,
        nClusters: 4,
        numEpochs: 3,
        randomState: 42,
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
        gridWidth: 2,
        gridHeight: 2,
        nClusters: 4,
        numEpochs: 3,
        randomState: 42,
      });
      await som.fit(data);

      const before = tf.memory().numTensors;
      await exportForVisualization(som, 'json');
      await exportForVisualization(som, 'csv');
      const after = tf.memory().numTensors;

      expect(after).toBe(before);

      som.dispose();
    });
  });

  describe('findOptimalClusters', () => {
    it('should not leak tensors with kmeans algorithm', async () => {
      const before = tf.memory().numTensors;

      await findOptimalClusters(data, {
        minClusters: 2,
        maxClusters: 3,
        algorithm: 'kmeans',
        algorithmParams: { randomState: 42, nInit: 1 },
      });

      const after = tf.memory().numTensors;
      expect(after).toBe(before);
    });
  });

  describe('Validation functions', () => {
    it('silhouetteScore should not leak tensors', () => {
      const before = tf.memory().numTensors;
      silhouetteScore(data, labels);
      const after = tf.memory().numTensors;
      expect(after).toBe(before);
    });

    it('daviesBouldin should not leak tensors', () => {
      const before = tf.memory().numTensors;
      daviesBouldin(data, labels);
      const after = tf.memory().numTensors;
      expect(after).toBe(before);
    });

    it('daviesBouldinEfficient should not leak tensors', () => {
      const before = tf.memory().numTensors;
      daviesBouldinEfficient(data, labels);
      const after = tf.memory().numTensors;
      expect(after).toBe(before);
    });

    it('calinskiHarabasz should not leak tensors', () => {
      const before = tf.memory().numTensors;
      calinskiHarabasz(data, labels);
      const after = tf.memory().numTensors;
      expect(after).toBe(before);
    });

    it('calinskiHarabaszEfficient should not leak tensors', () => {
      const before = tf.memory().numTensors;
      calinskiHarabaszEfficient(data, labels);
      const after = tf.memory().numTensors;
      expect(after).toBe(before);
    });
  });
});
