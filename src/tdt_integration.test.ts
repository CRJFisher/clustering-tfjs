import { PCA } from './decomposition/pca';
import { HDBSCAN } from './clustering/hdbscan';
import { KMeans } from './clustering/kmeans';
import { select_medoids } from './clustering/medoid_selection';
import { track_clusters, type TrackingState } from './clustering/cluster_tracking';
import { silhouette_score } from './validation/silhouette';
import { make_blobs } from './datasets/synthetic';

/**
 * End-to-end Topic Detection and Tracking pipeline integration test.
 *
 * Exercises the full bundle together on a browsing-timeline-shaped workload:
 * high-dimensional embedded events per time window are PCA-reduced, topics are
 * discovered with HDBSCAN (variable density, noise flagged), representatives are
 * extracted, topic quality is scored noise-aware, and topics are matched across
 * consecutive windows with track_clusters under stable lifelines.
 */
describe('TDT pipeline integration', () => {
  /** Build a window of high-dimensional embedded events with incidental noise. */
  function make_window(centers: number, seed: number, n: number): number[][] {
    const { X } = make_blobs({
      n_samples: n,
      n_features: 12,
      centers,
      cluster_std: 0.6,
      random_state: seed,
    });
    const rows = X.arraySync() as number[][];
    X.dispose();
    return rows;
  }

  it('detects topics, flags noise, and reduces dimensionality coherently', async () => {
    const window = make_window(3, 1, 90);

    // PCA pre-projection (1536-d embeddings → low-d before density clustering).
    const pca = new PCA({ n_components: 5, random_state: 0 });
    const reduced = pca.fit_transform(window);
    expect(reduced.length).toBe(window.length);
    expect(reduced[0].length).toBe(5);

    // HDBSCAN topic detection with noise.
    const hdbscan = new HDBSCAN({ min_cluster_size: 5, store_exemplars: true });
    const labels = await hdbscan.fit_predict(reduced);
    const topics = new Set(labels.filter((l) => l !== -1));
    expect(topics.size).toBeGreaterThanOrEqual(2);

    // Probabilities are well-formed.
    for (const p of hdbscan.probabilities_!) {
      expect(p).toBeGreaterThanOrEqual(0);
      expect(p).toBeLessThanOrEqual(1);
    }

    // Exemplars: one representative index per detected topic.
    expect(hdbscan.exemplar_indices_!.size).toBe(topics.size);

    // Noise-aware quality score is finite even with -1 present.
    const score = silhouette_score(reduced, labels);
    expect(Number.isFinite(score)).toBe(true);
  });

  it('tracks topics across consecutive windows with stable lifelines', async () => {
    // Two windows of the same 3 topics with mild drift between them.
    const w1 = make_window(3, 10, 90);
    const w2 = make_window(3, 11, 90);

    // Per-window topic representatives via KMeans centroids (stand-in snapshot
    // producer; tracking works with any representative vectors).
    const km1 = new KMeans({ n_clusters: 3, random_state: 42 });
    await km1.fit(w1);
    const km2 = new KMeans({ n_clusters: 3, random_state: 42 });
    await km2.fit(w2);

    const rep1 = km1.get_centroids();
    const rep2 = km2.get_centroids();

    // Frame 1 → 2.
    const r1 = track_clusters(rep1, rep2, { threshold: 0.5 });
    expect(r1.cost_matrix.length).toBe(3);
    expect(r1.cost_matrix[0].length).toBe(3);
    // Every transition carries a defined lifeline id.
    for (const t of r1.transitions) {
      expect(Number.isInteger(t.lifeline_id)).toBe(true);
    }

    // Frame 2 → 3 (reuse w1 representatives as a third drift target),
    // threading lifeline state forward.
    const km3 = new KMeans({ n_clusters: 3, random_state: 7 });
    await km3.fit(make_window(3, 12, 90));
    const rep3 = km3.get_centroids();

    const state: TrackingState = r1.state;
    const r2 = track_clusters(rep2, rep3, { threshold: 0.5 }, state);
    // Persisting topics keep their prior lifeline id across the frame.
    for (let j = 0; j < rep3.length; j++) {
      const i = r2.assignment[j];
      if (i >= 0) {
        expect(r2.state.lifelines[j]).toBe(state.lifelines[i]);
      }
    }

    km1.dispose();
    km2.dispose();
    km3.dispose();
  });

  it('medoid representatives interoperate with tracking (cosine geometry)', async () => {
    const w1 = make_window(3, 20, 75);
    const w2 = make_window(3, 21, 75);

    // Detect topics per window, then represent each by its medoid sample.
    const h1 = new HDBSCAN({ min_cluster_size: 5 });
    const l1 = await h1.fit_predict(w1);
    const k1 = new Set(l1.filter((l) => l !== -1)).size;
    const h2 = new HDBSCAN({ min_cluster_size: 5 });
    const l2 = await h2.fit_predict(w2);
    const k2 = new Set(l2.filter((l) => l !== -1)).size;
    expect(k1).toBeGreaterThanOrEqual(1);
    expect(k2).toBeGreaterThanOrEqual(1);

    const m1 = await select_medoids(w1, l1, k1, 'cosine');
    const m2 = await select_medoids(w2, l2, k2, 'cosine');

    const rep1 = Array.from(m1.indices)
      .filter((idx) => idx >= 0)
      .map((idx) => w1[idx]);
    const rep2 = Array.from(m2.indices)
      .filter((idx) => idx >= 0)
      .map((idx) => w2[idx]);

    const result = track_clusters(rep1, rep2, { threshold: 0.4 });
    // Tracking runs on real-sample representatives without error.
    expect(result.assignment.length).toBe(rep2.length);
    expect(result.transitions.length).toBeGreaterThan(0);
  });
});
