import { stored_nn_cluster, LinkageCriterion } from './linkage';

/**
 * Helper: build a flat Float64Array distance matrix from a 2D array.
 */
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

describe('storedNNCluster', () => {
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
    const merges = stored_nn_cluster(D, 3, 1, 'single');
    // 3 clusters down to 1 = 2 merges
    expect(merges.length).toBe(2);
  });

  it('first merge is always the closest pair (0,1)', () => {
    const linkages: LinkageCriterion[] = ['single', 'complete', 'average', 'ward'];
    for (const linkage of linkages) {
      const D = to_flat(base_matrix);
      const merges = stored_nn_cluster(D, 3, 1, linkage);
      // First merge should be clusters 0 and 1 (distance 2)
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
      const merges = stored_nn_cluster(D, 3, 1, linkage);
      expect(merges[1].distance).toBeCloseTo(expected_dist, 6);
    });
  });

  it('stops at the requested number of clusters', () => {
    const D = to_flat(base_matrix);
    const merges = stored_nn_cluster(D, 3, 2, 'single');
    // 3 clusters down to 2 = 1 merge
    expect(merges.length).toBe(1);
    expect(merges[0].cluster_a).toBe(0);
    expect(merges[0].cluster_b).toBe(1);
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
    const merges = stored_nn_cluster(D, 4, 1, 'single');

    expect(merges.length).toBe(3);

    // First merge: A+B (distance 1) — global minimum
    expect(merges[0].cluster_a).toBe(0);
    expect(merges[0].cluster_b).toBe(1);
    expect(merges[0].distance).toBeCloseTo(1, 6);

    // Second merge: C+D (distance 3) — next global minimum
    expect(merges[1].cluster_a).toBe(2);
    expect(merges[1].cluster_b).toBe(3);
    expect(merges[1].distance).toBeCloseTo(3, 6);

    // Third merge: {A,B}+{C,D} with single linkage = min(5,4,9,8) = 4
    expect(merges[2].distance).toBeCloseTo(4, 6);
  });

  it('cluster sizes are tracked correctly', () => {
    const D = to_flat(base_matrix);
    const merges = stored_nn_cluster(D, 3, 1, 'average');
    expect(merges[0].new_size).toBe(2); // merge of two singletons
    expect(merges[1].new_size).toBe(3); // merge size-2 with singleton
  });
});
