import { update_distance_matrix, DistanceMatrix, LinkageCriterion } from "../src/clustering/linkage";

function clone_matrix(D: DistanceMatrix): DistanceMatrix {
  return D.map((row) => [...row]);
}

describe("update_distance_matrix (linkage criteria)", () => {
  /**
   * Base distance matrix for three singleton clusters:
   *
   *    0   1   2
   * 0 [0,  2,  4]
   * 1 [2,  0,  6]
   * 2 [4,  6,  0]
   */
  const baseMatrix: DistanceMatrix = [
    [0, 2, 4],
    [2, 0, 6],
    [4, 6, 0],
  ];

  const clusterSizesBase = [1, 1, 1];

  const testCases: Array<{
    linkage: LinkageCriterion;
    expectedDist: number;
  }> = [
    { linkage: "single", expectedDist: 4 },
    { linkage: "complete", expectedDist: 6 },
    { linkage: "average", expectedDist: 5 },
    { linkage: "ward", expectedDist: Math.sqrt(100 / 3) },
  ];

  testCases.forEach(({ linkage, expectedDist }) => {
    it(`${linkage} linkage correctly updates the distance`, () => {
      const D = clone_matrix(baseMatrix);
      const sizes = [...clusterSizesBase];

      // merge clusters 0 and 1
      update_distance_matrix(D, sizes, 0, 1, linkage);

      expect(D.length).toBe(2);
      expect(D[0].length).toBe(2);

      // Distance between the new (0) cluster and old cluster 2 should match
      const actual = D[0][1];
      const tolerance = 1e-6;
      expect(Math.abs(actual - expectedDist)).toBeLessThan(tolerance);

      // Symmetry & zeros on diagonal
      expect(D[0][1]).toBeCloseTo(D[1][0], 6);
      expect(D[0][0]).toBe(0);
      expect(D[1][1]).toBe(0);

      // Cluster size bookkeeping
      expect(sizes).toEqual([2, 1]);
    });
  });
});

