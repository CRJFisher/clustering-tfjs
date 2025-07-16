import fs from "fs";
import path from "path";

import { AgglomerativeClustering, DataMatrix } from "../../src";

const FIXTURE_DIR = path.join(__dirname, "../fixtures/agglomerative");

function areLabelingsEquivalent(a: number[], b: number[]): boolean {
  if (a.length !== b.length) return false;
  const mapping = new Map<number, number>();
  for (let i = 0; i < a.length; i++) {
    const ai = a[i];
    const bi = b[i];
    if (!mapping.has(ai)) {
      mapping.set(ai, bi);
    } else if (mapping.get(ai) !== bi) {
      return false;
    }
  }
  return true;
}

describe("AgglomerativeClustering – reference parity with scikit-learn", () => {
  const files = fs
    .readdirSync(FIXTURE_DIR)
    .filter((f) => f.endsWith(".json"));

  if (files.length === 0) {
    console.warn("No reference fixtures found – skipping parity tests.");
    return;
  }

  for (const file of files) {
    const fixture = JSON.parse(
      fs.readFileSync(path.join(FIXTURE_DIR, file), "utf-8"),
    ) as {
      X: DataMatrix;
      params: { nClusters: number; linkage: string; metric: string };
      labels: number[];
    };

    it(`matches sklearn labels for ${file}`, async () => {
      const model = new AgglomerativeClustering({
        nClusters: fixture.params.nClusters,
        linkage: fixture.params.linkage as any,
        metric: fixture.params.metric as any,
      });

      const ours = (await model.fitPredict(fixture.X)) as number[];

      expect(areLabelingsEquivalent(ours, fixture.labels)).toBe(true);
    });
  }
});
