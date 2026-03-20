import * as tf from '../../src/tf-adapter';
import { SpectralClustering } from '../../src/clustering/spectral';

/**
 * Scale tests for spectral clustering with the Lanczos eigensolver.
 * Validates AC2: handles 5000+ samples without timeout.
 */
describe('Spectral clustering at scale', () => {
  // Generate well-separated blobs for easy clustering
  function generateBlobs(
    nSamples: number,
    nClusters: number,
    seed: number,
  ): { data: number[][]; labels: number[] } {
    const rng = require('../../src/utils/rng/index').make_random_stream(seed);
    const data: number[][] = [];
    const labels: number[] = [];
    const samplesPerCluster = Math.floor(nSamples / nClusters);

    for (let c = 0; c < nClusters; c++) {
      // Place cluster centers far apart
      const cx = c * 10;
      const cy = c * 10;

      for (let i = 0; i < samplesPerCluster; i++) {
        // Box-Muller for Gaussian noise
        const u1 = rng.rand();
        const u2 = rng.rand();
        const z0 = Math.sqrt(-2 * Math.log(u1 + 1e-10)) * Math.cos(2 * Math.PI * u2);
        const z1 = Math.sqrt(-2 * Math.log(u1 + 1e-10)) * Math.sin(2 * Math.PI * u2);
        data.push([cx + z0, cy + z1]);
        labels.push(c);
      }
    }

    return { data, labels };
  }

  it('handles 1000 samples efficiently', async () => {
    const { data } = generateBlobs(1000, 3, 42);

    const tensorsBefore = tf.memory().numTensors;
    const start = performance.now();

    const sc = new SpectralClustering({
      nClusters: 3,
      affinity: 'rbf',
      randomState: 42,
    });

    await sc.fit(data);
    const elapsed = performance.now() - start;

    expect(sc.labels_).not.toBeNull();
    expect(new Set(sc.labels_!).size).toBe(3);
    console.log(`  n=1000: ${elapsed.toFixed(0)}ms`);

    sc.dispose();

    // Verify no significant tensor leaks
    const tensorsAfter = tf.memory().numTensors;
    expect(tensorsAfter - tensorsBefore).toBeLessThan(100);
  }, 60000);

  it('handles 5000+ samples without timeout (AC2)', async () => {
    const { data } = generateBlobs(5000, 3, 42);

    const start = performance.now();

    const sc = new SpectralClustering({
      nClusters: 3,
      affinity: 'rbf',
      randomState: 42,
    });

    await sc.fit(data);
    const elapsed = performance.now() - start;

    expect(sc.labels_).not.toBeNull();
    expect(new Set(sc.labels_!).size).toBe(3);
    console.log(`  n=5000: ${elapsed.toFixed(0)}ms`);

    // Should complete well within 120 seconds
    expect(elapsed).toBeLessThan(120000);

    sc.dispose();
  }, 120000);
});
