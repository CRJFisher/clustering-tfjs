import * as tf from "@tensorflow/tfjs-node";
import { SpectralClustering } from "./src/clustering/spectral";
import * as fs from "fs";
import * as path from "path";

const FIXTURE_DIR = path.join(__dirname, "test/fixtures/spectral");

// Adjusted Rand Index calculation
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

async function testNInitImprovement() {
  // The failing RBF tests
  const failingTests = [
    "circles_n2_rbf.json",
    "moons_n2_rbf.json", 
    "circles_n3_rbf.json",
    "moons_n3_rbf.json",
    "blobs_n3_rbf.json"
  ];

  console.log("Testing different nInit values on failing RBF tests:\n");
  console.log("=".repeat(60) + "\n");

  for (const testFile of failingTests) {
    const fixture = JSON.parse(
      fs.readFileSync(path.join(FIXTURE_DIR, testFile), "utf-8")
    );

    console.log(`Test: ${testFile}`);
    console.log("-".repeat(40));
    
    const X = tf.tensor2d(fixture.X);
    const y_true = fixture.labels;
    const params = {
      nClusters: fixture.params.nClusters,
      affinity: fixture.params.affinity as "rbf",
      gamma: fixture.params.gamma,
      randomState: fixture.params.randomState
    };

    // Test different nInit values
    const nInitValues = [10, 50, 100, 200];
    const results: number[] = [];
    
    for (const nInit of nInitValues) {
      const sc = new SpectralClustering({
        ...params,
        nInit
      });
      const labels = await sc.fitPredict(X) as number[];
      const ari = adjustedRandIndex(y_true, labels);
      results.push(ari);
      console.log(`nInit=${nInit.toString().padStart(3)}: ARI = ${ari.toFixed(4)}`);
    }

    // Check improvement
    const baseline = results[0];
    const best = Math.max(...results);
    const improvement = best - baseline;
    console.log(`\nBaseline (nInit=10): ${baseline.toFixed(4)}`);
    console.log(`Best result:         ${best.toFixed(4)} (improvement: ${improvement > 0 ? '+' : ''}${improvement.toFixed(4)})`);
    
    console.log();
    
    // Cleanup
    X.dispose();
  }
}

testNInitImprovement().catch(console.error);