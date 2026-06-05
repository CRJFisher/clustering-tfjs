import { performance } from 'perf_hooks';
import { AgglomerativeClustering } from './agglomerative';
import { make_blobs } from '../datasets/synthetic';

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

    console.log(`5000 samples, ward linkage: ${elapsed.toFixed(2)}s`);
  }, 120_000);
});
