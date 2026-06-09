import { SpectralClustering } from '..';

/**
 * SpectralClustering with `affinity='cosine'` builds its similarity graph from
 * cosine affinity and produces a valid embedding and labels. Direction-
 * dominated, magnitude-noisy data is the intended use case.
 */
describe("SpectralClustering – affinity='cosine'", () => {
  it('produces a valid two-cluster labeling on direction-separated data', async () => {
    // Two angular clusters at differing magnitudes; cosine ignores magnitude.
    const cluster_a = [
      [1, 0.05],
      [2, 0.1],
      [3, -0.05],
      [4, 0.0],
    ];
    const cluster_b = [
      [0.05, 1],
      [0.1, 2],
      [-0.05, 3],
      [0.0, 4],
    ];
    const X = [...cluster_a, ...cluster_b];

    const model = new SpectralClustering({
      n_clusters: 2,
      affinity: 'cosine',
      random_state: 0,
    });
    const labels = await model.fit_predict(X);

    expect(labels.length).toBe(X.length);
    // Exactly two clusters present.
    expect(new Set(labels).size).toBe(2);
    // Each angular group is internally consistent.
    const a_labels = labels.slice(0, 4);
    const b_labels = labels.slice(4);
    expect(new Set(a_labels).size).toBe(1);
    expect(new Set(b_labels).size).toBe(1);
    expect(a_labels[0]).not.toBe(b_labels[0]);

    model.dispose();
  });

  it('accepts cosine as a valid affinity string', () => {
    expect(
      () => new SpectralClustering({ n_clusters: 2, affinity: 'cosine' }),
    ).not.toThrow();
  });
});
