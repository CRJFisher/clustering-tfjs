import fs from "fs";
import path from "path";

import { SpectralClustering, DataMatrix } from "../../src";

const FIXTURE_DIR = path.join(__dirname, "../fixtures/spectral");

// Adjusted Rand Index helper – measures similarity independent of label permutation.
function adjustedRandIndex(labelsA: number[], labelsB: number[]): number {
  if (labelsA.length !== labelsB.length) {
    throw new Error("Label arrays must have same length");
  }

  const n = labelsA.length;
  const labelToIndexA = new Map<number, number>();
  const labelToIndexB = new Map<number, number>();

  let nextA = 0;
  let nextB = 0;

  const contingency: number[][] = [];

  for (let i = 0; i < n; i++) {
    const a = labelsA[i];
    const b = labelsB[i];

    if (!labelToIndexA.has(a)) {
      labelToIndexA.set(a, nextA++);
      contingency.push([]);
    }
    const idxA = labelToIndexA.get(a)!;

    if (!labelToIndexB.has(b)) {
      labelToIndexB.set(b, nextB++);
      // Ensure all rows have the same length
      for (const row of contingency) {
        while (row.length < nextB) row.push(0);
      }
    }
    const idxB = labelToIndexB.get(b)!;

    // Ensure this row has enough columns
    while (contingency[idxA].length <= idxB) {
      contingency[idxA].push(0);
    }

    contingency[idxA][idxB] = (contingency[idxA][idxB] || 0) + 1;
  }

  const ai = contingency.map((row) => row.reduce((s, v) => s + v, 0));
  const bj = contingency[0].map((_, j) => contingency.reduce((s, row) => s + row[j], 0));

  const comb2 = (x: number): number => (x * (x - 1)) / 2;

  let sumComb = 0;
  for (const row of contingency) {
    for (const val of row) sumComb += comb2(val);
  }

  const sumAi = ai.reduce((s, v) => s + comb2(v), 0);
  const sumBj = bj.reduce((s, v) => s + comb2(v), 0);

  const expected = (sumAi * sumBj) / comb2(n);
  const max = (sumAi + sumBj) / 2;

  if (max === expected) return 0;
  return (sumComb - expected) / (max - expected);
}

describe("SpectralClustering – reference parity with scikit-learn", () => {
  if (!fs.existsSync(FIXTURE_DIR)) {
    it("skipped – no spectral fixtures dir present", () => {
      expect(true).toBe(true);
    });
    return;
  }

  const files = fs
    .readdirSync(FIXTURE_DIR)
    .filter((f) => f.endsWith(".json"));

  if (files.length === 0) {
    it("skipped – no spectral reference fixtures present", () => {
      expect(true).toBe(true);
    });
    return;
  }

  for (const file of files) {
    const fixture = JSON.parse(
      fs.readFileSync(path.join(FIXTURE_DIR, file), "utf-8"),
    ) as {
      X: DataMatrix;
      params: {
        nClusters: number;
        affinity: string;
        gamma?: number;
        nNeighbors?: number;
        randomState: number;
      };
      labels: number[];
    };

    it(`matches sklearn labels for ${file}`, async () => {
      const ctorParams: any = {
        nClusters: fixture.params.nClusters,
        affinity: fixture.params.affinity,
        randomState: fixture.params.randomState,
      };
      if (fixture.params.gamma !== undefined && fixture.params.gamma !== null) {
        ctorParams.gamma = fixture.params.gamma;
      }
      if (fixture.params.nNeighbors !== undefined && fixture.params.nNeighbors !== null) {
        ctorParams.nNeighbors = fixture.params.nNeighbors;
      }

      const model = new SpectralClustering(ctorParams);

      const ours = (await model.fitPredict(fixture.X)) as number[];

      const ari = adjustedRandIndex(ours, fixture.labels);
      expect(ari).toBeGreaterThanOrEqual(0.95);
    }, 20000); // allow generous timeout – eigen decomposition can be slow
  }
});
