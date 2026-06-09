import fs from 'fs';
import path from 'path';

import {
  build_condensation_tree,
  condense_hierarchy,
  excess_of_mass,
  extract_labels,
  compute_stability,
} from './condensation_tree';
import type { CondensedEdge } from './condensation_tree';
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

  it('returns an empty tree for an empty hierarchy and labels a lone point noise', () => {
    expect(condense_hierarchy([], 1, 2)).toEqual([]);
    expect(excess_of_mass([], 1).size).toBe(0);
    const { labels, probabilities } = extract_labels([], new Set(), 1);
    expect(labels).toEqual([-1]);
    expect(probabilities).toEqual([0]);
  });

  it('maps coincident points (merge distance 0) to infinite lambda and probability 1', () => {
    // Two coincident points merge at distance 0 -> lambda = Infinity.
    const tree = condense_hierarchy([[0, 1, 0, 2]], 2, 2);
    expect(tree).toHaveLength(2);
    for (const e of tree) {
      expect(e.lambda_val).toBe(Number.POSITIVE_INFINITY);
    }

    // Leaf selection on a cluster-less tree falls back to the root; with
    // single-cluster output allowed, both points join it at probability 1.
    const selected = excess_of_mass(tree, 2, {
      cluster_selection_method: 'leaf',
    });
    expect(selected).toEqual(new Set([2]));
    const { labels, probabilities } = extract_labels(tree, selected, 2, true);
    expect(labels).toEqual([0, 0]);
    expect(probabilities).toEqual([1, 1]);

    // Without allow_single_cluster the root is not assignable: all noise.
    const strict = extract_labels(tree, selected, 2, false);
    expect(strict.labels).toEqual([-1, -1]);
  });
});

describe('condensation_tree – structural edge cases on hand-built trees', () => {
  it('treats a missing parent birth as zero stability contribution', () => {
    // Parent 99 never appears as a child and is not the root, so it has no
    // recorded birth; its stability accumulates from λ = 0.
    const tree: CondensedEdge[] = [
      { parent: 99, child: 0, lambda_val: 2, child_size: 1 },
    ];
    expect(compute_stability(tree, 2).get(99)).toBe(2);
  });

  it('keeps a λ=0-born leaf cluster: its birth distance is infinite', () => {
    // A cluster edge at λ = 0 means birth distance 1/λ = Infinity, which can
    // never be below epsilon, so the leaf survives the epsilon merge.
    const tree: CondensedEdge[] = [
      { parent: 4, child: 5, lambda_val: 0, child_size: 4 },
      { parent: 5, child: 0, lambda_val: 1, child_size: 1 },
      { parent: 5, child: 1, lambda_val: 1, child_size: 1 },
      { parent: 5, child: 2, lambda_val: 1, child_size: 1 },
      { parent: 5, child: 3, lambda_val: 1, child_size: 1 },
    ];
    const selected = excess_of_mass(tree, 4, {
      cluster_selection_method: 'leaf',
      cluster_selection_epsilon: 10,
    });
    expect(selected).toEqual(new Set([5]));
  });

  it('labels a point noise when its cluster chain is orphaned', () => {
    // Point 0 falls out of cluster 5, which is unselected and has no parent
    // edge: the climb dead-ends and the point stays noise.
    const tree: CondensedEdge[] = [
      { parent: 5, child: 0, lambda_val: 2, child_size: 1 },
    ];
    const { labels } = extract_labels(tree, new Set(), 2);
    expect(labels).toEqual([-1, -1]);
  });

  it('assigns probability 1 when a cluster dies at λ=0', () => {
    // Every edge of cluster 5 sits at λ = 0, so its death λ is 0 and the
    // probability ratio degenerates; members get full membership.
    const tree: CondensedEdge[] = [
      { parent: 5, child: 0, lambda_val: 0, child_size: 1 },
    ];
    const { labels, probabilities } = extract_labels(tree, new Set([5]), 2);
    expect(labels).toEqual([0, -1]);
    expect(probabilities[0]).toBe(1);
  });
});

/**
 * Epsilon-merge semantics on a hand-built condensed tree, pinning every
 * `traverse_upwards` branch directly. Tree over n=8 points (root id 8):
 *
 *   8 ─λ=1.0─> 9  ─λ=2.0─> 11 ─λ=4.0─> points 0,1
 *               └─λ=2.0─> 12 ─λ=4.0─> points 2,3
 *   8 ─λ=1.0─> 10 ─λ=1.5─> points 4,5,6,7
 *
 * Birth distances (1/λ): clusters 9 and 10 at 1.0; leaves 11 and 12 at 0.5.
 */
describe('condensation_tree – epsilon merge on a hand-built tree', () => {
  const tree: CondensedEdge[] = [
    { parent: 8, child: 9, lambda_val: 1, child_size: 4 },
    { parent: 8, child: 10, lambda_val: 1, child_size: 4 },
    { parent: 9, child: 11, lambda_val: 2, child_size: 2 },
    { parent: 9, child: 12, lambda_val: 2, child_size: 2 },
    { parent: 11, child: 0, lambda_val: 4, child_size: 1 },
    { parent: 11, child: 1, lambda_val: 4, child_size: 1 },
    { parent: 12, child: 2, lambda_val: 4, child_size: 1 },
    { parent: 12, child: 3, lambda_val: 4, child_size: 1 },
    { parent: 10, child: 4, lambda_val: 1.5, child_size: 1 },
    { parent: 10, child: 5, lambda_val: 1.5, child_size: 1 },
    { parent: 10, child: 6, lambda_val: 1.5, child_size: 1 },
    { parent: 10, child: 7, lambda_val: 1.5, child_size: 1 },
  ];
  const n = 8;

  it('selects the raw leaves when epsilon is 0', () => {
    const selected = excess_of_mass(tree, n, {
      cluster_selection_method: 'leaf',
    });
    expect(selected).toEqual(new Set([10, 11, 12]));
  });

  it('climbs one level to the first ancestor born coarser than epsilon', () => {
    // Leaves 11/12 are born at 0.5 < 0.8; their parent 9 is born at 1.0 >= 0.8,
    // so the climb stops there. Leaf 10 (born 1.0) is kept as-is, and 12 is
    // skipped as already-processed once 9 absorbs it.
    const selected = excess_of_mass(tree, n, {
      cluster_selection_method: 'leaf',
      cluster_selection_epsilon: 0.8,
    });
    expect(selected).toEqual(new Set([9, 10]));

    const { labels } = extract_labels(tree, selected, n);
    // Points 0-3 route to 9 (their leaves are unselected), 4-7 to 10.
    expect(labels.slice(0, 4)).toEqual([0, 0, 0, 0]);
    expect(labels.slice(4)).toEqual([1, 1, 1, 1]);
  });

  it('recurses past ancestors still finer than epsilon and stops below the root', () => {
    // With epsilon 1.5 even cluster 9 (born 1.0) is too fine; its parent is
    // the root, so the climb returns the last sub-root cluster.
    const selected = excess_of_mass(tree, n, {
      cluster_selection_method: 'leaf',
      cluster_selection_epsilon: 1.5,
    });
    expect(selected).toEqual(new Set([9, 10]));
  });

  it('merges into the root when allow_single_cluster is set', () => {
    const selected = excess_of_mass(tree, n, {
      cluster_selection_method: 'leaf',
      cluster_selection_epsilon: 1.5,
      allow_single_cluster: true,
    });
    expect(selected).toEqual(new Set([n]));

    const single = extract_labels(tree, selected, n, true);
    expect(single.labels).toEqual(new Array(n).fill(0));
    for (const p of single.probabilities) expect(p).toBe(1);
  });

  it('applies the epsilon merge to the eom selection too', () => {
    // EOM picks {10, 11, 12} (children of 9 beat it: 4 + 4 > 4); epsilon 0.8
    // then coarsens 11/12 into 9 exactly as in leaf mode.
    expect(
      excess_of_mass(tree, n, { cluster_selection_method: 'eom' }),
    ).toEqual(new Set([10, 11, 12]));
    expect(
      excess_of_mass(tree, n, {
        cluster_selection_method: 'eom',
        cluster_selection_epsilon: 0.8,
      }),
    ).toEqual(new Set([9, 10]));
  });

  it('eom keeps a parent and drops its descendants when the parent is more stable', () => {
    // Shrink the grandchildren's persistence so 9 beats them:
    // stability(11) = stability(12) = (2.5-2)*2 = 1 each; stability(9) = 4.
    const weak = tree.map((e) =>
      e.child_size === 1 && (e.parent === 11 || e.parent === 12)
        ? { ...e, lambda_val: 2.5 }
        : e,
    );
    const selected = excess_of_mass(weak, n, {
      cluster_selection_method: 'eom',
    });
    expect(selected).toEqual(new Set([9, 10]));
  });
});
