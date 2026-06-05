import * as tf from '../backend/adapter';
import { lanczos_smallest_eigenpairs } from './lanczos';
import { improved_jacobi_eigen } from './improved';
import { normalised_laplacian } from '../graph/laplacian';
import { compute_rbf_affinity } from '../graph/affinity';

/**
 * Benchmark: Lanczos vs Jacobi eigensolver.
 * Validates AC3: >10x speedup over Jacobi for n>500.
 */
describe('Eigensolver benchmark: Lanczos vs Jacobi', () => {
  function generate_symmetric_laplacian(n: number, seed: number): tf.Tensor2D {
    const rng = require('../random').make_random_stream(seed);
    const data: number[][] = [];
    const n_clusters = 3;
    const samples_per_cluster = Math.floor(n / n_clusters);

    for (let c = 0; c < n_clusters; c++) {
      const cx = c * 10;
      const cy = c * 10;
      for (let i = 0; i < samples_per_cluster; i++) {
        const u1 = rng.rand();
        const u2 = rng.rand();
        const z0 = Math.sqrt(-2 * Math.log(u1 + 1e-10)) * Math.cos(2 * Math.PI * u2);
        const z1 = Math.sqrt(-2 * Math.log(u1 + 1e-10)) * Math.sin(2 * Math.PI * u2);
        data.push([cx + z0, cy + z1]);
      }
    }

    const points = tf.tensor2d(data);
    const affinity = compute_rbf_affinity(points);
    points.dispose();

    const lap = normalised_laplacian(affinity) as tf.Tensor2D;
    affinity.dispose();

    return lap;
  }

  function time_it(fn: () => void, runs: number = 3): number {
    const times: number[] = [];
    for (let i = 0; i < runs; i++) {
      const start = performance.now();
      fn();
      times.push(performance.now() - start);
    }
    times.sort((a, b) => a - b);
    return times[Math.floor(times.length / 2)]; // median
  }

  it('Lanczos is faster than Jacobi for n=500', () => {
    const k = 3;
    const lap = generate_symmetric_laplacian(500, 42);

    const jacobi_time = time_it(() => {
      improved_jacobi_eigen(lap, { is_psd: true, max_iterations: 3000, tolerance: 1e-14 });
    }, 1);

    const lanczos_time = time_it(() => {
      lanczos_smallest_eigenpairs(lap, k, { is_psd: true, random_seed: 42 });
    }, 3);

    lap.dispose();

    const speedup = jacobi_time / lanczos_time;
    console.log(`  n=500: Jacobi=${jacobi_time.toFixed(0)}ms, Lanczos=${lanczos_time.toFixed(0)}ms, speedup=${speedup.toFixed(1)}x`);

    // AC3: >10x speedup for n>500
    expect(speedup).toBeGreaterThan(10);
  }, 120000);

  it('Lanczos produces correct eigenvalues compared to Jacobi for n=200', () => {
    const k = 3;
    const lap = generate_symmetric_laplacian(200, 42);

    const jacobi_result = improved_jacobi_eigen(lap, {
      is_psd: true,
      max_iterations: 3000,
      tolerance: 1e-14,
    });

    const lanczos_result = lanczos_smallest_eigenpairs(lap, k, {
      is_psd: true,
      random_seed: 42,
    });

    lap.dispose();

    // Compare the k smallest eigenvalues
    for (let i = 0; i < k; i++) {
      expect(lanczos_result.eigenvalues[i]).toBeCloseTo(
        jacobi_result.eigenvalues[i],
        2,
      );
    }
  }, 60000);

  it('Lanczos handles n=1000 in reasonable time', () => {
    const k = 3;
    const lap = generate_symmetric_laplacian(1000, 42);

    const start = performance.now();
    const result = lanczos_smallest_eigenpairs(lap, k, { is_psd: true, random_seed: 42 });
    const elapsed = performance.now() - start;

    lap.dispose();

    console.log(`  n=1000 Lanczos: ${elapsed.toFixed(0)}ms`);
    expect(result.eigenvalues).toHaveLength(k);
    expect(result.eigenvalues[0]).toBeCloseTo(0, 2); // Connected graph → λ₁ ≈ 0
    expect(elapsed).toBeLessThan(30000); // Should be well under 30s
  }, 60000);
});
