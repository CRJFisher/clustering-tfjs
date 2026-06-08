import { SpectralClustering } from './spectral';

describe('SpectralClustering sparse nearest-neighbors path', () => {
  it('stores nearest-neighbor affinity sparsely without dense affinity state', async () => {
    const X = [
      [0, 0],
      [0.1, 0],
      [0, 0.1],
      [10, 10],
      [10.1, 10],
      [10, 10.1],
    ];
    const model = new SpectralClustering({
      n_clusters: 2,
      affinity: 'nearest_neighbors',
      n_neighbors: 3,
      random_state: 42,
      capture_debug_info: true,
    });

    await model.fit(X);

    expect(model.labels_).toHaveLength(X.length);
    expect(model.affinity_matrix_).toBeNull();
    expect(model.sparse_affinity_matrix_).not.toBeNull();
    expect(model.get_debug_info()?.affinity_stats?.shape).toEqual([6, 6]);
    expect(model.get_debug_info()?.affinity_stats?.nnz).toBeLessThan(36);
  });

  it('allows large sample counts past max_samples for nearest_neighbors only', async () => {
    const X = Array.from({ length: 12 }, (_, i) => [i, i % 2]);
    const sparse_model = new SpectralClustering({
      n_clusters: 2,
      affinity: 'nearest_neighbors',
      n_neighbors: 2,
      max_samples: 5,
      random_state: 42,
    });
    await expect(sparse_model.fit(X)).resolves.toBeUndefined();

    const dense_model = new SpectralClustering({
      n_clusters: 2,
      affinity: 'rbf',
      max_samples: 5,
    });
    await expect(dense_model.fit(X)).rejects.toThrow(/exceeds the maximum/);
  });
});
