import fs from 'fs';
import path from 'path';

import { HDBSCAN } from '..';
import type { HDBSCANParams } from './types';
import * as tf from '../../test_support/tensorflow_helper';
import {
  alignment_agreement,
  labels_equivalent_with_noise,
} from '../../test_support/label_agreement';

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
  /** Whether the fixture's MST edge weights are all distinct (see generator). */
  tie_free: boolean;
  /** Raw diagnostic behind tie_free; asserted generator-side, not here. */
  min_mst_gap: number;
  X?: number[][];
  distance_matrix?: number[][];
}

function load_fixtures(filter: (name: string) => boolean = () => true): {
  file: string;
  fixture: HdbscanFixture;
}[] {
  return fs
    .readdirSync(FIXTURE_DIR)
    .filter((f) => f.endsWith('.json') && filter(f))
    .map((file) => ({
      file,
      fixture: JSON.parse(
        fs.readFileSync(path.join(FIXTURE_DIR, file), 'utf-8'),
      ) as HdbscanFixture,
    }));
}

function fit_input(fixture: HdbscanFixture): number[][] {
  return fixture.params.metric === 'precomputed'
    ? fixture.distance_matrix!
    : fixture.X!;
}

function fixture_params(fixture: HdbscanFixture): Partial<HDBSCANParams> {
  const params: Partial<HDBSCANParams> = {
    min_cluster_size: fixture.params.min_cluster_size,
    cluster_selection_method: fixture.params.cluster_selection_method,
    cluster_selection_epsilon: fixture.params.cluster_selection_epsilon,
    metric: fixture.params.metric,
  };
  if (fixture.params.min_samples != null) {
    params.min_samples = fixture.params.min_samples;
  }
  return params;
}

/**
 * End-to-end parity with scikit-learn. The condensed-tree + EOM core is
 * validated bit-exactly against sklearn's own hierarchy in
 * `condensation_tree.test.ts`; here the full pipeline (including
 * minimum-spanning-tree construction) runs from raw input.
 *
 * The assertion is tiered by the fixture's `tie_free` flag:
 *
 * - **Tie-free** (all MST edge weights distinct): the MST — and hence the
 *   whole flat clustering — is unique, so labels must match exactly (up to
 *   cluster-id permutation) and probabilities per-point to 1e-6.
 * - **Tie-bound**: numpy's unstable argsort orders tied mutual-reachability
 *   weights differently than Prim's algorithm, shifting a few boundary
 *   points between equally valid hierarchies. Cluster count stays exact,
 *   label agreement >= 0.95 under optimal alignment, and the per-fixture
 *   probability MAE is bounded by 0.16. The whole pipeline is deterministic
 *   float64, so the observed MAEs are stable; the worst fixture (precomputed
 *   cosine, whose saturated distances tie heavily) sits at 0.150 and every
 *   other fixture at or below 0.077. The bound is that observed maximum plus
 *   deliberate slack — exceeding it means the fixtures were regenerated under
 *   a different scikit-learn version, which calls for re-measuring, not
 *   loosening.
 */
describe('HDBSCAN – parity with scikit-learn', () => {
  for (const { file, fixture } of load_fixtures()) {
    it(`matches labels and probabilities for ${file}`, async () => {
      const model = new HDBSCAN(fixture_params(fixture));
      const labels = await model.fit_predict(fit_input(fixture));

      expect(labels.length).toBe(fixture.labels.length);
      const probs = model.probabilities_!;
      for (const p of probs) {
        expect(p).toBeGreaterThanOrEqual(0);
        expect(p).toBeLessThanOrEqual(1 + 1e-9);
      }

      if (fixture.tie_free) {
        expect(labels_equivalent_with_noise(labels, fixture.labels)).toBe(true);
        for (let i = 0; i < probs.length; i++) {
          expect(probs[i]).toBeCloseTo(fixture.probabilities[i], 6);
        }
        return;
      }

      // Same number of clusters and consistent noise vs. sklearn.
      const mine_clusters = new Set(labels.filter((l) => l !== -1)).size;
      const sk_clusters = new Set(fixture.labels.filter((l) => l !== -1)).size;
      expect(mine_clusters).toBe(sk_clusters);
      expect(alignment_agreement(labels, fixture.labels)).toBeGreaterThanOrEqual(
        0.95,
      );
      let mae = 0;
      for (let i = 0; i < probs.length; i++) {
        mae += Math.abs(probs[i] - fixture.probabilities[i]);
      }
      expect(mae / probs.length).toBeLessThanOrEqual(0.16);
    });
  }
});

/**
 * Degenerate inputs, pinned to scikit-learn output. Both come back all-noise:
 * an all-noise dataset because no density peak clears min_cluster_size, and a
 * single dense blob because candidate clusters are created in sibling pairs,
 * so with `allow_single_cluster` unsupported a root-only tree labels nothing.
 */
describe('HDBSCAN – degenerate cases match scikit-learn', () => {
  const degenerate = load_fixtures(
    (f) => f.startsWith('allnoise_') || f.startsWith('single_blob_'),
  );

  it('has fixtures for both degenerate datasets in both modes', () => {
    expect(degenerate.length).toBe(4);
  });

  for (const { file, fixture } of degenerate) {
    it(`labels every sample noise with probability 0 for ${file}`, async () => {
      expect(fixture.labels.every((l) => l === -1)).toBe(true);

      const model = new HDBSCAN(fixture_params(fixture));
      const labels = await model.fit_predict(fit_input(fixture));
      expect(labels).toEqual(fixture.labels);
      expect(model.probabilities_).toEqual(fixture.probabilities);
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

  it('rejects empty input', async () => {
    const model = new HDBSCAN({ min_cluster_size: 4 });
    await expect(model.fit([])).rejects.toThrow('at least one sample');
  });

  it('preserves prior fitted state when fit throws on invalid input', async () => {
    const model = new HDBSCAN({ min_cluster_size: 4, store_exemplars: true });
    await model.fit(X);

    const saved_labels = model.labels_!.slice();
    const saved_probs = model.probabilities_!.slice();
    const saved_exemplars = new Map(model.exemplar_indices_!);

    // Empty input: n === 0 guard fires before dispose(), state must survive.
    await expect(model.fit([])).rejects.toThrow('at least one sample');
    expect(model.labels_).toEqual(saved_labels);
    expect(model.probabilities_).toEqual(saved_probs);
    expect(model.exemplar_indices_).toEqual(saved_exemplars);

    // Ragged input: distance_matrix throws before dispose() is reached,
    // state must also survive.
    await expect(
      model.fit([
        [1, 2],
        [1, 2, 3],
      ]),
    ).rejects.toThrow('rectangular');
    expect(model.labels_).toEqual(saved_labels);
    expect(model.probabilities_).toEqual(saved_probs);
    expect(model.exemplar_indices_).toEqual(saved_exemplars);
  });

  it('labels a single sample noise (deliberate deviation: sklearn raises)', async () => {
    const with_ex = new HDBSCAN({ min_cluster_size: 4, store_exemplars: true });
    await with_ex.fit([[1, 2]]);
    expect(with_ex.labels_).toEqual([-1]);
    expect(with_ex.probabilities_).toEqual([0]);
    expect(with_ex.exemplar_indices_).toEqual(new Map());

    const without = new HDBSCAN({ min_cluster_size: 4 });
    await without.fit([[1, 2]]);
    expect(without.labels_).toEqual([-1]);
    expect(without.exemplar_indices_).toBeNull();
  });

  it('rejects ragged input rows', async () => {
    const model = new HDBSCAN({ min_cluster_size: 4 });
    await expect(
      model.fit([
        [0, 1],
        [1, 0, 2],
      ]),
    ).rejects.toThrow('rectangular');
  });

  it('rejects a non-square precomputed distance matrix', async () => {
    const model = new HDBSCAN({ min_cluster_size: 4, metric: 'precomputed' });
    await expect(
      model.fit([
        [0, 1, 2],
        [1, 0, 1],
      ]),
    ).rejects.toThrow('square');
  });

  it('accepts tensor input for native and precomputed metrics', async () => {
    const from_array = new HDBSCAN({ min_cluster_size: 4 });
    const array_labels = await from_array.fit_predict(X);

    const points = tf.tensor2d(X);
    const from_tensor = new HDBSCAN({ min_cluster_size: 4 });
    const tensor_labels = await from_tensor.fit_predict(points);
    points.dispose();
    expect(tensor_labels).toEqual(array_labels);

    const n = X.length;
    const D: number[][] = Array.from({ length: n }, (_v, i) =>
      Array.from({ length: n }, (_w, j) => {
        let s = 0;
        for (let d = 0; d < X[i].length; d++) {
          const diff = X[i][d] - X[j][d];
          s += diff * diff;
        }
        return Math.sqrt(s);
      }),
    );
    const D_tensor = tf.tensor2d(D);
    const precomputed = new HDBSCAN({
      min_cluster_size: 4,
      metric: 'precomputed',
    });
    const precomputed_labels = await precomputed.fit_predict(D_tensor);
    D_tensor.dispose();
    expect(precomputed_labels).toEqual(array_labels);
  });

  it('clusters under the manhattan metric', async () => {
    const model = new HDBSCAN({ min_cluster_size: 4, metric: 'manhattan' });
    const labels = await model.fit_predict(X);
    expect(labels.length).toBe(X.length);
    expect(new Set(labels.filter((l) => l !== -1)).size).toBe(2);
  });

  it('clamps min_samples to n (deliberate deviation: sklearn raises)', async () => {
    const model = new HDBSCAN({ min_cluster_size: 4, min_samples: 100 });
    const labels = await model.fit_predict(X);
    expect(labels.length).toBe(X.length);
  });

  it('uses default params when constructed without arguments', async () => {
    const model = new HDBSCAN();
    // Default min_cluster_size is 5; both blobs have 5 members.
    const labels = await model.fit_predict(X);
    expect(new Set(labels.filter((l) => l !== -1)).size).toBe(2);
  });

  it('selects exemplars deterministically across fits', async () => {
    const a = new HDBSCAN({ min_cluster_size: 4, store_exemplars: true });
    const b = new HDBSCAN({ min_cluster_size: 4, store_exemplars: true });
    await a.fit(X);
    await b.fit(X);
    expect(a.exemplar_indices_).toEqual(b.exemplar_indices_);
  });
});

/**
 * The parameter sweeps required for sklearn-grounded coverage: at least three
 * `min_samples` values and three `cluster_selection_epsilon` values, each in
 * both `eom` and `leaf` modes. Locks the fixture inventory itself so a
 * regenerated set cannot silently drop the sweeps.
 */
describe('HDBSCAN – fixture sweep inventory', () => {
  const all = load_fixtures();

  for (const method of ['eom', 'leaf'] as const) {
    it(`sweeps min_samples over >= 3 values in ${method} mode`, () => {
      const values = new Set(
        all
          .filter(
            ({ fixture }) =>
              fixture.params.cluster_selection_method === method &&
              fixture.params.min_samples != null,
          )
          .map(({ fixture }) => fixture.params.min_samples),
      );
      expect(values.size).toBeGreaterThanOrEqual(3);
    });

    it(`sweeps cluster_selection_epsilon over >= 3 values in ${method} mode`, () => {
      const values = new Set(
        all
          .filter(
            ({ fixture }) =>
              fixture.params.cluster_selection_method === method &&
              fixture.params.cluster_selection_epsilon > 0,
          )
          .map(({ fixture }) => fixture.params.cluster_selection_epsilon),
      );
      expect(values.size).toBeGreaterThanOrEqual(3);
    });
  }
});
