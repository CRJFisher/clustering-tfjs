import fs from "fs";
import path from "path";

import {
  AgglomerativeClustering,
  AgglomerativeClusteringParams,
  DataMatrix,
} from "..";

// Use path relative to project root for fixtures
const FIXTURE_DIR = path.join(process.cwd(), "__fixtures__", "agglomerative");

function are_labelings_equivalent(a: number[], b: number[]): boolean {
  if (a.length !== b.length) return false;
  const forward_mapping = new Map<number, number>();
  const reverse_mapping = new Map<number, number>();
  for (let i = 0; i < a.length; i++) {
    const ai = a[i];
    const bi = b[i];
    // Check forward mapping: a -> b
    if (forward_mapping.has(ai)) {
      if (forward_mapping.get(ai) !== bi) return false;
    } else {
      forward_mapping.set(ai, bi);
    }
    // Check reverse mapping: b -> a
    if (reverse_mapping.has(bi)) {
      if (reverse_mapping.get(bi) !== ai) return false;
    } else {
      reverse_mapping.set(bi, ai);
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
      params: { n_clusters: number; linkage: string; metric: string };
      labels: number[];
    };

    it(`matches sklearn labels for ${file}`, async () => {
      const model = new AgglomerativeClustering({
        n_clusters: fixture.params.n_clusters,
        linkage: fixture.params.linkage as AgglomerativeClusteringParams['linkage'],
        metric: fixture.params.metric as AgglomerativeClusteringParams['metric'],
      });

      const ours = await model.fit_predict(fixture.X);

      expect(are_labelings_equivalent(ours, fixture.labels)).toBe(true);
    });
  }
});
