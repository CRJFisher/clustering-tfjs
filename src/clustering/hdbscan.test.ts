import fs from 'fs';
import path from 'path';

import { HDBSCAN } from '..';
import type { HDBSCANParams } from './types';

const FIXTURE_DIR = path.join(process.cwd(), '__fixtures__', 'hdbscan');

interface HdbscanFixture {
  name: string;
  params: {
    min_cluster_size: number;
    min_samples: number | null;
    cluster_selection_method: 'eom' | 'leaf';
    cluster_selection_epsilon: number;
    metric: 'euclidean' | 'manhattan' | 'precomputed';
  };
  labels: number[];
  probabilities: number[];
  X?: number[][];
  distance_matrix?: number[][];
}

/**
 * Cluster-assignment agreement under the optimal cluster-id alignment, with
 * noise (-1) treated as its own label.
 */
function alignment_agreement(mine: number[], sk: number[]): number {
  const pairs = new Map<string, number>();
  for (let i = 0; i < mine.length; i++) {
    const k = `${sk[i]}|${mine[i]}`;
    pairs.set(k, (pairs.get(k) ?? 0) + 1);
  }
  const map = new Map<number, number>();
  for (const s of new Set(sk)) {
    let best = -99;
    let bc = -1;
    for (const m of new Set(mine)) {
      const c = pairs.get(`${s}|${m}`) ?? 0;
      if (c > bc) {
        bc = c;
        best = m;
      }
    }
    map.set(s, best);
  }
  let ok = 0;
  for (let i = 0; i < mine.length; i++) if (map.get(sk[i]) === mine[i]) ok++;
  return ok / mine.length;
}

/**
 * End-to-end parity with scikit-learn. The condensed-tree + EOM core is
 * validated bit-exactly against sklearn's own hierarchy in
 * `condensation_tree.test.ts`; here the full pipeline (including
 * minimum-spanning-tree construction) is compared with tolerance, because
 * mutual-reachability weight ties are ordered differently across
 * implementations (numpy's unstable argsort), shifting a few boundary points.
 */
describe('HDBSCAN – parity with scikit-learn', () => {
  const files = fs.readdirSync(FIXTURE_DIR).filter((f) => f.endsWith('.json'));

  for (const file of files) {
    const fixture = JSON.parse(
      fs.readFileSync(path.join(FIXTURE_DIR, file), 'utf-8'),
    ) as HdbscanFixture;

    it(`matches labels and probabilities for ${file}`, async () => {
      const params: Partial<HDBSCANParams> = {
        min_cluster_size: fixture.params.min_cluster_size,
        cluster_selection_method: fixture.params.cluster_selection_method,
        cluster_selection_epsilon: fixture.params.cluster_selection_epsilon,
        metric: fixture.params.metric,
      };
      if (fixture.params.min_samples != null) {
        params.min_samples = fixture.params.min_samples;
      }

      const model = new HDBSCAN(params);
      const data =
        fixture.params.metric === 'precomputed'
          ? fixture.distance_matrix!
          : fixture.X!;
      const labels = await model.fit_predict(data);

      expect(labels.length).toBe(fixture.labels.length);
      // Same number of clusters and consistent noise vs. sklearn.
      const mine_clusters = new Set(labels.filter((l) => l !== -1)).size;
      const sk_clusters = new Set(fixture.labels.filter((l) => l !== -1)).size;
      expect(mine_clusters).toBe(sk_clusters);
      expect(alignment_agreement(labels, fixture.labels)).toBeGreaterThanOrEqual(
        0.95,
      );

      // Probabilities are in range and close in aggregate (tie-sensitive).
      const probs = model.probabilities_!;
      let mae = 0;
      for (let i = 0; i < probs.length; i++) {
        expect(probs[i]).toBeGreaterThanOrEqual(0);
        expect(probs[i]).toBeLessThanOrEqual(1 + 1e-9);
        mae += Math.abs(probs[i] - fixture.probabilities[i]);
      }
      expect(mae / probs.length).toBeLessThanOrEqual(0.2);
    });
  }
});

describe('HDBSCAN – API surface', () => {
  const X = [
    [0, 0],
    [0.1, 0.1],
    [0.0, 0.2],
    [0.2, 0.0],
    [0.1, 0.0],
    [10, 10],
    [10.1, 10.1],
    [10.0, 10.2],
    [10.2, 10.0],
    [10.1, 10.0],
    [50, 50], // isolated noise point
  ];

  it('emits -1 for noise and probabilities in [0,1]', async () => {
    const model = new HDBSCAN({ min_cluster_size: 4 });
    const labels = await model.fit_predict(X);
    expect(labels).toContain(-1);
    for (const p of model.probabilities_!) {
      expect(p).toBeGreaterThanOrEqual(0);
      expect(p).toBeLessThanOrEqual(1);
    }
  });

  it('populates exemplar_indices_ only when store_exemplars is set', async () => {
    const without = new HDBSCAN({ min_cluster_size: 4 });
    await without.fit(X);
    expect(without.exemplar_indices_).toBeNull();

    const with_ex = new HDBSCAN({ min_cluster_size: 4, store_exemplars: true });
    await with_ex.fit(X);
    expect(with_ex.exemplar_indices_).not.toBeNull();
    const labels = new Set(with_ex.labels_!.filter((l) => l !== -1));
    expect(with_ex.exemplar_indices_!.size).toBe(labels.size);
    for (const [, idx] of with_ex.exemplar_indices_!) {
      expect(idx).toBeGreaterThanOrEqual(0);
      expect(idx).toBeLessThan(X.length);
    }
  });

  it('accepts a precomputed distance matrix', async () => {
    const n = X.length;
    const D: number[][] = Array.from({ length: n }, () =>
      new Array<number>(n).fill(0),
    );
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        let s = 0;
        for (let d = 0; d < X[i].length; d++) {
          const diff = X[i][d] - X[j][d];
          s += diff * diff;
        }
        D[i][j] = Math.sqrt(s);
      }
    }
    const model = new HDBSCAN({ min_cluster_size: 4, metric: 'precomputed' });
    const labels = await model.fit_predict(D);
    expect(labels.length).toBe(n);
  });

  it('exposes no predict method and no n_clusters param', () => {
    const model = new HDBSCAN({ min_cluster_size: 4 });
    expect('predict' in model).toBe(false);
    expect('n_clusters' in model.params).toBe(false);
  });

  it('validates params', () => {
    expect(() => new HDBSCAN({ min_cluster_size: 1 })).toThrow();
    expect(() => new HDBSCAN({ min_samples: 0 })).toThrow();
    // @ts-expect-error invalid method
    expect(() => new HDBSCAN({ cluster_selection_method: 'bad' })).toThrow();
  });

  it('dispose resets fitted state', async () => {
    const model = new HDBSCAN({ min_cluster_size: 4 });
    await model.fit(X);
    model.dispose();
    expect(model.labels_).toBeNull();
    expect(model.probabilities_).toBeNull();
  });
});
