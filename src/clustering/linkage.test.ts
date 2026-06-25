import { AgglomerativeClustering } from './agglomerative';
import { nn_chain_cluster, LinkageCriterion } from './linkage';

function to_flat(D2d: number[][]): Float64Array {
  const n = D2d.length;
  const flat = new Float64Array(n * n);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      flat[i * n + j] = D2d[i][j];
    }
  }
  return flat;
}

describe('nn_chain_cluster', () => {
  /**
   * Base distance matrix for three singleton clusters:
   *
   *    0   1   2
   * 0 [0,  2,  4]
   * 1 [2,  0,  6]
   * 2 [4,  6,  0]
   *
   * Closest pair is (0,1) with distance 2.
   * After merging 0+1 the remaining distance to cluster 2 depends on linkage.
   */
  const base_matrix: number[][] = [
    [0, 2, 4],
    [2, 0, 6],
    [4, 6, 0],
  ];

  it('produces the correct number of merges', () => {
    const D = to_flat(base_matrix);
    const merges = nn_chain_cluster(D, 3, 'single');
    // NN-chain always builds the full tree: 3 samples -> 2 merges.
    expect(merges.length).toBe(2);
  });

  it('first merge is always the closest pair (0,1)', () => {
    const linkages: LinkageCriterion[] = ['single', 'complete', 'average', 'ward'];
    for (const linkage of linkages) {
      const D = to_flat(base_matrix);
      const merges = nn_chain_cluster(D, 3, linkage);
      expect(merges[0].cluster_a).toBe(0);
      expect(merges[0].cluster_b).toBe(1);
      expect(merges[0].distance).toBeCloseTo(2, 6);
    }
  });

  const expected_second_merge_dist: Array<{
    linkage: LinkageCriterion;
    expected_dist: number;
  }> = [
    { linkage: 'single', expected_dist: 4 },     // min(4, 6)
    { linkage: 'complete', expected_dist: 6 },    // max(4, 6)
    { linkage: 'average', expected_dist: 5 },     // (1*4 + 1*6) / 2
    { linkage: 'ward', expected_dist: Math.sqrt(100 / 3) }, // Ward formula
  ];

  expected_second_merge_dist.forEach(({ linkage, expected_dist }) => {
    it(`${linkage} linkage: second merge distance is correct`, () => {
      const D = to_flat(base_matrix);
      const merges = nn_chain_cluster(D, 3, linkage);
      expect(merges[1].distance).toBeCloseTo(expected_dist, 6);
    });
  });

  it('cuts labels at the requested number of clusters', async () => {
    const model = new AgglomerativeClustering({
      n_clusters: 2,
      linkage: 'single',
      metric: 'precomputed',
    });
    const labels = await model.fit_predict(base_matrix);
    expect(new Set(labels).size).toBe(2);
    expect(model.children_).toEqual([[0, 1]]);
  });

  it('handles a 4-point dataset correctly with single linkage', () => {
    // Distances:
    //   A-B=1, A-C=5, A-D=9
    //   B-C=4, B-D=8
    //   C-D=3
    const D2d = [
      [0, 1, 5, 9],
      [1, 0, 4, 8],
      [5, 4, 0, 3],
      [9, 8, 3, 0],
    ];
    const D = to_flat(D2d);
    const merges = nn_chain_cluster(D, 4, 'single');

    expect(merges.length).toBe(3);

    expect(merges[0].cluster_a).toBe(0);
    expect(merges[0].cluster_b).toBe(1);
    expect(merges[0].distance).toBeCloseTo(1, 6);

    expect(merges[1].cluster_a).toBe(2);
    expect(merges[1].cluster_b).toBe(3);
    expect(merges[1].distance).toBeCloseTo(3, 6);

    // Third merge: {A,B}+{C,D} with single linkage = min(5,4,9,8) = 4
    expect(merges[2].distance).toBeCloseTo(4, 6);
  });

  it('cluster sizes are tracked correctly', () => {
    const D = to_flat(base_matrix);
    const merges = nn_chain_cluster(D, 3, 'average');
    expect(merges[0].new_size).toBe(2);
    expect(merges[1].new_size).toBe(3);
  });

  it('n=2 produces a single merge with correct distance and size', () => {
    const D = new Float64Array([0, 3.5, 3.5, 0]);
    const merges = nn_chain_cluster(D, 2, 'single');
    expect(merges.length).toBe(1);
    expect(merges[0].cluster_a).toBe(0);
    expect(merges[0].cluster_b).toBe(1);
    expect(merges[0].distance).toBeCloseTo(3.5, 10);
    expect(merges[0].new_size).toBe(2);
  });

  it('cluster_a < cluster_b in every merge record for all linkages', () => {
    const D2d = [
      [0, 1, 5, 9],
      [1, 0, 4, 8],
      [5, 4, 0, 3],
      [9, 8, 3, 0],
    ];
    const linkages: LinkageCriterion[] = ['single', 'complete', 'average', 'ward'];
    for (const linkage of linkages) {
      const merges = nn_chain_cluster(to_flat(D2d), 4, linkage);
      for (const m of merges) {
        expect(m.cluster_a).toBeLessThan(m.cluster_b);
      }
    }
  });

  it('output is sorted: merge distances are non-decreasing', () => {
    const D2d = [
      [0, 1, 5, 9],
      [1, 0, 4, 8],
      [5, 4, 0, 3],
      [9, 8, 3, 0],
    ];
    const linkages: LinkageCriterion[] = ['single', 'complete', 'average', 'ward'];
    for (const linkage of linkages) {
      const merges = nn_chain_cluster(to_flat(D2d), 4, linkage);
      for (let i = 1; i < merges.length; i++) {
        expect(merges[i].distance).toBeGreaterThanOrEqual(merges[i - 1].distance);
      }
    }
  });
});

/**
 * Ward tie-breaking parity with scikit-learn on a symmetric integer grid.
 *
 * A regular grid has many exactly-tied distances. The dendrogram is only
 * reproducible if the Lance–Williams update rounds bit-identically to scipy
 * and distances are stored as float64. Expected partitions are taken from
 * `sklearn.cluster.AgglomerativeClustering(linkage='ward')` and compared
 * permutation-invariantly via group membership.
 */
describe('AgglomerativeClustering – ward tie parity with scikit-learn', () => {
  function grid_points(side: number): number[][] {
    const pts: number[][] = [];
    for (let i = 0; i < side; i++) {
      for (let j = 0; j < side; j++) pts.push([i, j]);
    }
    return pts;
  }

  function partition_signature(labels: number[]): number[][] {
    const groups = new Map<number, number[]>();
    labels.forEach((lab, idx) => {
      const g = groups.get(lab);
      if (g) g.push(idx);
      else groups.set(lab, [idx]);
    });
    return [...groups.values()]
      .map((g) => g.sort((a, b) => a - b))
      .sort((a, b) => a[0] - b[0]);
  }

  // Expected groups straight from sklearn AgglomerativeClustering(ward).
  const cases: Array<{
    side: number;
    k: number;
    expected: number[][];
  }> = [
    { side: 4, k: 2, expected: [
      [0, 1, 4, 5, 8, 9, 12, 13],
      [2, 3, 6, 7, 10, 11, 14, 15],
    ] },
    { side: 4, k: 3, expected: [
      [0, 1, 4, 5, 8, 9, 12, 13],
      [2, 3, 6, 7],
      [10, 11, 14, 15],
    ] },
    { side: 4, k: 4, expected: [
      [0, 1, 4, 5],
      [2, 3, 6, 7],
      [8, 9, 12, 13],
      [10, 11, 14, 15],
    ] },
    { side: 5, k: 2, expected: [
      [0, 1, 5, 6, 10, 11, 15, 16, 20, 21],
      [2, 3, 4, 7, 8, 9, 12, 13, 14, 17, 18, 19, 22, 23, 24],
    ] },
    { side: 5, k: 3, expected: [
      [0, 1, 5, 6, 10, 11, 15, 16, 20, 21],
      [2, 3, 4, 7, 8, 9],
      [12, 13, 14, 17, 18, 19, 22, 23, 24],
    ] },
  ];

  cases.forEach(({ side, k, expected }) => {
    it(`matches sklearn ward on a ${side}x${side} grid at k=${k}`, async () => {
      const model = new AgglomerativeClustering({
        n_clusters: k,
        linkage: 'ward',
      });
      const labels = await model.fit_predict(grid_points(side));
      expect(partition_signature(labels)).toEqual(
        expected.map((g) => [...g].sort((a, b) => a - b)).sort((a, b) => a[0] - b[0]),
      );
    });
  });
});
