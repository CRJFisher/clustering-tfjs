import { performance } from 'perf_hooks';
import { AgglomerativeClustering } from './agglomerative';
import { nn_chain_cluster } from './linkage';
import { make_blobs } from '../datasets/synthetic';

function make_hub_distance_matrix(n: number): Float64Array {
  const D = new Float64Array(n * n);

  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      const distance =
        i === 0 || j === 0
          ? 1 + Math.min(i, j) * 1e-9
          : 2 + Math.abs(i - j) * 1e-6;
      D[i * n + j] = distance;
      D[j * n + i] = distance;
    }
  }

  return D;
}

describe('Agglomerative clustering performance', () => {
  it('1000 samples completes in under 5 seconds', async () => {
    const { X } = make_blobs({
      n_samples: 1000,
      n_features: 2,
      centers: 5,
      random_state: 42,
    });

    const data = (await X.array()) as number[][];
    X.dispose();

    const model = new AgglomerativeClustering({
      n_clusters: 5,
      linkage: 'ward',
    });

    const start = performance.now();
    await model.fit(data);
    const elapsed = (performance.now() - start) / 1000;

    expect(model.labels_).not.toBeNull();
    expect((model.labels_ as number[]).length).toBe(1000);
    expect(elapsed).toBeLessThan(5);

    console.log(`1000 samples, ward linkage: ${elapsed.toFixed(2)}s`);
  });

  it('5000 samples completes (benchmark)', async () => {
    const { X } = make_blobs({
      n_samples: 5000,
      n_features: 2,
      centers: 5,
      random_state: 42,
    });

    const data = (await X.array()) as number[][];
    X.dispose();

    const model = new AgglomerativeClustering({
      n_clusters: 5,
      linkage: 'ward',
    });

    const start = performance.now();
    await model.fit(data);
    const elapsed = (performance.now() - start) / 1000;

    expect(model.labels_).not.toBeNull();
    expect((model.labels_ as number[]).length).toBe(5000);
    // Loose upper bound: an O(n³) regression would take many minutes at
    // n=5000, so a generous 60s ceiling fails loudly on a cubic blowup while
    // leaving ample headroom for the typical O(n²) run on slow CI.
    expect(elapsed).toBeLessThan(60);

    console.log(`5000 samples, ward linkage: ${elapsed.toFixed(2)}s`);
  }, 120_000);

  it('merge loop avoids cubic blowup on hub-shaped nearest neighbors', () => {
    function time_merge_loop(n: number): number {
      const D = make_hub_distance_matrix(n);
      const start = performance.now();
      const merges = nn_chain_cluster(D, n, 'single');
      expect(merges.length).toBe(n - 1);
      return performance.now() - start;
    }

    // Warm up the JIT before comparing scaling.
    time_merge_loop(64);

    const small_elapsed = time_merge_loop(200);
    const large_elapsed = time_merge_loop(400);
    const ratio = large_elapsed / Math.max(small_elapsed, 1e-9);

    expect(ratio).toBeLessThan(8);
    console.log(
      `hub NN-chain scaling 200->400: ${small_elapsed.toFixed(2)}ms -> ${large_elapsed.toFixed(2)}ms (${ratio.toFixed(2)}x)`,
    );
  });
});
