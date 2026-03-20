import { storedNNCluster, LinkageCriterion } from '../src/clustering/linkage';

/**
 * Helper: build a flat Float64Array distance matrix from a 2D array.
 */
function toFlat(D2d: number[][]): Float64Array {
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
  const baseMatrix: number[][] = [
    [0, 2, 4],
    [2, 0, 6],
    [4, 6, 0],
  ];

  it('produces the correct number of merges', () => {
    const D = toFlat(baseMatrix);
    const merges = storedNNCluster(D, 3, 1, 'single');
    // 3 clusters down to 1 = 2 merges
    expect(merges.length).toBe(2);
  });

  it('first merge is always the closest pair (0,1)', () => {
    const linkages: LinkageCriterion[] = ['single', 'complete', 'average', 'ward'];
    for (const linkage of linkages) {
      const D = toFlat(baseMatrix);
      const merges = storedNNCluster(D, 3, 1, linkage);
      // First merge should be clusters 0 and 1 (distance 2)
      expect(merges[0].clusterA).toBe(0);
      expect(merges[0].clusterB).toBe(1);
      expect(merges[0].distance).toBeCloseTo(2, 6);
    }
  });

  const expectedSecondMergeDist: Array<{
    linkage: LinkageCriterion;
    expectedDist: number;
  }> = [
    { linkage: 'single', expectedDist: 4 },     // min(4, 6)
    { linkage: 'complete', expectedDist: 6 },    // max(4, 6)
    { linkage: 'average', expectedDist: 5 },     // (1*4 + 1*6) / 2
    { linkage: 'ward', expectedDist: Math.sqrt(100 / 3) }, // Ward formula
  ];

  expectedSecondMergeDist.forEach(({ linkage, expectedDist }) => {
    it(`${linkage} linkage: second merge distance is correct`, () => {
      const D = toFlat(baseMatrix);
      const merges = storedNNCluster(D, 3, 1, linkage);
      expect(merges[1].distance).toBeCloseTo(expectedDist, 6);
    });
  });

  it('stops at the requested number of clusters', () => {
    const D = toFlat(baseMatrix);
    const merges = storedNNCluster(D, 3, 2, 'single');
    // 3 clusters down to 2 = 1 merge
    expect(merges.length).toBe(1);
    expect(merges[0].clusterA).toBe(0);
    expect(merges[0].clusterB).toBe(1);
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
    const D = toFlat(D2d);
    const merges = storedNNCluster(D, 4, 1, 'single');

    expect(merges.length).toBe(3);

    // First merge: A+B (distance 1) — global minimum
    expect(merges[0].clusterA).toBe(0);
    expect(merges[0].clusterB).toBe(1);
    expect(merges[0].distance).toBeCloseTo(1, 6);

    // Second merge: C+D (distance 3) — next global minimum
    expect(merges[1].clusterA).toBe(2);
    expect(merges[1].clusterB).toBe(3);
    expect(merges[1].distance).toBeCloseTo(3, 6);

    // Third merge: {A,B}+{C,D} with single linkage = min(5,4,9,8) = 4
    expect(merges[2].distance).toBeCloseTo(4, 6);
  });

  it('cluster sizes are tracked correctly', () => {
    const D = toFlat(baseMatrix);
    const merges = storedNNCluster(D, 3, 1, 'average');
    expect(merges[0].newSize).toBe(2); // merge of two singletons
    expect(merges[1].newSize).toBe(3); // merge size-2 with singleton
  });
});
