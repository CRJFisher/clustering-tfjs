import * as tf from '../../src/backend/adapter';
import { SpectralClustering } from '../../src/clustering/spectral';
import { make_random_stream } from '../../src/random';

/**
 * Scale tests for spectral clustering with the Lanczos eigensolver.
 * Validates AC2: handles 5000+ samples without timeout.
 */
describe('Spectral clustering at scale', () => {
  // Generate well-separated blobs for easy clustering
  function generate_blobs(
    n_samples: number,
    n_clusters: number,
    seed: number,
  ): { data: number[][]; labels: number[] } {
    const rng = make_random_stream(seed);
    const data: number[][] = [];
    const labels: number[] = [];
    const samples_per_cluster = Math.floor(n_samples / n_clusters);

    for (let c = 0; c < n_clusters; c++) {
      // Place cluster centers far apart
      const cx = c * 10;
      const cy = c * 10;

      for (let i = 0; i < samples_per_cluster; i++) {
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
    const { data } = generate_blobs(1000, 3, 42);

    const tensors_before = tf.memory().numTensors;
    const start = performance.now();

    const sc = new SpectralClustering({
      n_clusters: 3,
      affinity: 'rbf',
      random_state: 42,
    });

    await sc.fit(data);
    const elapsed = performance.now() - start;

    expect(sc.labels_).not.toBeNull();
    expect(new Set(sc.labels_!).size).toBe(3);
    console.log(`  n=1000: ${elapsed.toFixed(0)}ms`);

    sc.dispose();

    // Verify no significant tensor leaks
    const tensors_after = tf.memory().numTensors;
    expect(tensors_after - tensors_before).toBeLessThan(100);
  }, 60000);

  it('handles 5000+ samples without timeout (AC2)', async () => {
    const { data } = generate_blobs(5000, 3, 42);

    const start = performance.now();

    const sc = new SpectralClustering({
      n_clusters: 3,
      affinity: 'rbf',
      random_state: 42,
    });

    await sc.fit(data);
    const elapsed = performance.now() - start;

    expect(sc.labels_).not.toBeNull();
    expect(new Set(sc.labels_!).size).toBe(3);
    console.log(`  n=5000: ${elapsed.toFixed(0)}ms`);

    // Should complete well within 3 minutes (generous for slow CI runners)
    expect(elapsed).toBeLessThan(180_000);

    sc.dispose();
  }, 240_000);
});
