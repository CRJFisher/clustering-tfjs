import fs from 'fs';
import path from 'path';

import {
  build_condensation_tree,
  condense_hierarchy,
  excess_of_mass,
  extract_labels,
  compute_stability,
} from './condensation_tree';
import { minimum_spanning_tree } from './minimum_spanning_tree';
import { mutual_reachability } from './mutual_reachability';
import { kdistance } from '../distance/kdistance';

const FIXTURE_DIR = path.join(process.cwd(), '__fixtures__', 'hdbscan');

function metric_matrix(X: number[][], metric: string): number[][] {
  const n = X.length;
  const D: number[][] = Array.from({ length: n }, () =>
    new Array<number>(n).fill(0),
  );
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      let s = 0;
      for (let d = 0; d < X[i].length; d++) {
        const diff = X[i][d] - X[j][d];
        s += metric === 'manhattan' ? Math.abs(diff) : diff * diff;
      }
      const dist = metric === 'manhattan' ? s : Math.sqrt(s);
      D[i][j] = dist;
      D[j][i] = dist;
    }
  }
  return D;
}

function euclidean_matrix(X: number[][]): number[][] {
  return metric_matrix(X, 'euclidean');
}

function labels_equivalent_with_noise(a: number[], b: number[]): boolean {
  if (a.length !== b.length) return false;
  const fwd = new Map<number, number>();
  const rev = new Map<number, number>();
  for (let i = 0; i < a.length; i++) {
    if ((a[i] === -1) !== (b[i] === -1)) return false;
    if (a[i] === -1) continue;
    if (fwd.has(a[i])) {
      if (fwd.get(a[i]) !== b[i]) return false;
    } else fwd.set(a[i], b[i]);
    if (rev.has(b[i])) {
      if (rev.get(b[i]) !== a[i]) return false;
    } else rev.set(b[i], a[i]);
  }
  return true;
}

interface HdbscanFixture {
  params: {
    min_cluster_size: number;
    min_samples: number | null;
    cluster_selection_method: 'eom' | 'leaf';
    cluster_selection_epsilon: number;
    metric: string;
  };
  labels: number[];
  probabilities: number[];
  single_linkage_tree: number[][];
  X?: number[][];
  distance_matrix?: number[][];
}

const files = fs.readdirSync(FIXTURE_DIR).filter((f) => f.endsWith('.json'));

/**
 * The condensed tree + EOM selection is validated for BIT-EXACT parity by
 * feeding it scikit-learn's own single-linkage hierarchy. This isolates the
 * module under test from minimum-spanning-tree tie-ordering, which differs
 * across implementations (numpy's unstable argsort) and is exercised
 * separately, with tolerance, in the end-to-end estimator test.
 */
describe('condensation_tree – exact parity on sklearn hierarchy', () => {
  for (const file of files) {
    const fixture = JSON.parse(
      fs.readFileSync(path.join(FIXTURE_DIR, file), 'utf-8'),
    ) as HdbscanFixture;

    it(`reproduces sklearn labels and probabilities for ${file}`, () => {
      const n = fixture.single_linkage_tree.length + 1;
      const tree = condense_hierarchy(
        fixture.single_linkage_tree,
        n,
        fixture.params.min_cluster_size,
      );
      const selected = excess_of_mass(tree, n, {
        cluster_selection_method: fixture.params.cluster_selection_method,
        cluster_selection_epsilon: fixture.params.cluster_selection_epsilon,
      });
      const { labels, probabilities } = extract_labels(tree, selected, n);

      // Exact labelling (up to cluster-id permutation, noise consistent).
      expect(labels_equivalent_with_noise(labels, fixture.labels)).toBe(true);
      // Exact probabilities.
      for (let i = 0; i < n; i++) {
        expect(probabilities[i]).toBeCloseTo(fixture.probabilities[i], 6);
      }
    });
  }
});

/**
 * End-to-end from raw data, the flat clustering matches sklearn closely. Small
 * boundary disagreements are expected where mutual-reachability weights tie and
 * the two implementations order them differently.
 */
describe('condensation_tree – end-to-end agreement from raw data', () => {
  function agreement(mine: number[], sk: number[]): number {
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

  for (const file of files) {
    const fixture = JSON.parse(
      fs.readFileSync(path.join(FIXTURE_DIR, file), 'utf-8'),
    ) as HdbscanFixture;

    it(`agrees with sklearn from raw data for ${file}`, () => {
      const D =
        fixture.params.metric === 'precomputed'
          ? fixture.distance_matrix!
          : metric_matrix(fixture.X!, fixture.params.metric);
      const n = D.length;
      const ms =
        fixture.params.min_samples ?? fixture.params.min_cluster_size;
      const nd = D.map((row) => [...row].sort((a, b) => a - b));
      const core = kdistance(nd, Math.min(ms, n));
      const mreach = mutual_reachability(D, core);
      const mst = minimum_spanning_tree(mreach);
      const tree = build_condensation_tree(
        mst,
        n,
        fixture.params.min_cluster_size,
      );
      const selected = excess_of_mass(tree, n, {
        cluster_selection_method: fixture.params.cluster_selection_method,
        cluster_selection_epsilon: fixture.params.cluster_selection_epsilon,
      });
      const { labels } = extract_labels(tree, selected, n);

      const mine_clusters = new Set(labels.filter((l) => l !== -1)).size;
      const sk_clusters = new Set(fixture.labels.filter((l) => l !== -1)).size;
      expect(mine_clusters).toBe(sk_clusters);
      expect(agreement(labels, fixture.labels)).toBeGreaterThanOrEqual(0.95);
    });
  }
});

describe('condensation_tree – degenerate inputs', () => {
  function cluster(D: number[][], mcs: number, ms: number, method: 'eom' | 'leaf') {
    const n = D.length;
    const nd = D.map((row) => [...row].sort((a, b) => a - b));
    const core = kdistance(nd, Math.min(ms, n));
    const mst = minimum_spanning_tree(mutual_reachability(D, core));
    const tree = build_condensation_tree(mst, n, mcs);
    const selected = excess_of_mass(tree, n, {
      cluster_selection_method: method,
    });
    return {
      labels: extract_labels(tree, selected, n).labels,
      stabilities: [...compute_stability(tree, n).values()],
    };
  }

  it('handles all-equidistant points with finite labels and stability', () => {
    const n = 6;
    const D: number[][] = Array.from({ length: n }, (_v, i) =>
      Array.from({ length: n }, (_w, j) => (i === j ? 0 : 1)),
    );
    const { labels, stabilities } = cluster(D, 3, 3, 'eom');
    expect(labels.length).toBe(n);
    for (const l of labels) expect(Number.isFinite(l)).toBe(true);
    for (const s of stabilities) expect(Number.isNaN(s)).toBe(false);
  });

  it('resolves a single dense cluster without NaN', () => {
    const X = [
      [0, 0],
      [0.1, 0.0],
      [0.0, 0.1],
      [0.1, 0.1],
      [0.05, 0.05],
      [0.02, 0.08],
    ];
    const { labels, stabilities } = cluster(euclidean_matrix(X), 2, 2, 'eom');
    expect(labels.length).toBe(X.length);
    for (const l of labels) expect(Number.isFinite(l)).toBe(true);
    for (const s of stabilities) expect(Number.isNaN(s)).toBe(false);
  });

  it('leaf selection yields valid labels', () => {
    const X = [
      [0, 0],
      [0.1, 0],
      [0.0, 0.1],
      [5, 5],
      [5.1, 5],
      [5, 5.1],
    ];
    const { labels } = cluster(euclidean_matrix(X), 2, 2, 'leaf');
    expect(labels.length).toBe(X.length);
    expect(new Set(labels.filter((l) => l !== -1)).size).toBeGreaterThanOrEqual(1);
  });
});
