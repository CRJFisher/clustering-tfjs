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

  it('fit_with_intermediate_steps matches fit() for nearest_neighbors when n > max_samples', async () => {
    const X = Array.from({ length: 12 }, (_, i) => [i, i % 2]);
    const params = {
      n_clusters: 2,
      affinity: 'nearest_neighbors' as const,
      n_neighbors: 2,
      max_samples: 5,
      random_state: 42,
    };

    const model_fit = new SpectralClustering(params);
    await model_fit.fit(X);

    const model_steps = new SpectralClustering(params);
    const steps = await model_steps.fit_with_intermediate_steps(X);

    // Both bypass the max_samples guard and succeed
    expect(model_fit.sparse_affinity_matrix_).not.toBeNull();
    expect(model_steps.sparse_affinity_matrix_).not.toBeNull();

    // Both produce sparse affinity with the same shape
    expect(model_fit.sparse_affinity_matrix_!.rows).toBe(X.length);
    expect(model_steps.sparse_affinity_matrix_!.rows).toBe(X.length);
    expect(model_fit.sparse_affinity_matrix_!.cols).toBe(X.length);
    expect(model_steps.sparse_affinity_matrix_!.cols).toBe(X.length);

    // Both compute identical sparse affinity values (same computation path)
    expect(Array.from(model_steps.sparse_affinity_matrix_!.indices)).toEqual(
      Array.from(model_fit.sparse_affinity_matrix_!.indices),
    );
    expect(Array.from(model_steps.sparse_affinity_matrix_!.data)).toEqual(
      Array.from(model_fit.sparse_affinity_matrix_!.data),
    );

    // Dense affinity_matrix_ is null for both (sparse path)
    expect(model_fit.affinity_matrix_).toBeNull();
    expect(model_steps.affinity_matrix_).toBeNull();

    // The affinity returned in steps has the correct dense shape
    expect(steps.affinity.shape).toEqual([X.length, X.length]);

    model_fit.dispose();
    steps.affinity.dispose();
    steps.laplacian.laplacian.dispose();
    steps.laplacian.degrees?.dispose();
    steps.laplacian.sqrt_degrees?.dispose();
    steps.embedding.embedding.dispose();
    steps.embedding.eigenvalues.dispose();
    steps.embedding.raw_eigenvectors?.dispose();
    model_steps.dispose();
  });
});
