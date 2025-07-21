import * as tf from "@tensorflow/tfjs-node";
import { SpectralClustering } from "./src/clustering/spectral";
import { compute_rbf_affinity } from "./src/utils/affinity";
import * as fs from "fs";

async function testKMeansImprovement() {
  // Load fixture
  const fixture = JSON.parse(fs.readFileSync("./test/fixtures/spectral/circles_n2_rbf.json", "utf8"));
  const X = tf.tensor2d(fixture.X);
  const y_true = fixture.labels;
  
  console.log("Testing k-means improvements on circles_n2_rbf:\n");
  
  // Test 1: Current implementation with default nInit=10
  console.log("1. Current implementation (nInit=10):");
  const sc1 = new SpectralClustering({
    nClusters: 2,
    affinity: "rbf",
    gamma: 1.0,
    randomState: 42
  });
  
  const labels1 = await sc1.fitPredict(X);
  const ari1 = calculateARI(y_true, labels1 as number[]);
  console.log(`   ARI: ${ari1.toFixed(4)}`);
  
  // Test 2: Try with higher nInit
  console.log("\n2. With higher nInit=50:");
  const sc2 = new SpectralClustering({
    nClusters: 2,
    affinity: "rbf",
    gamma: 1.0,
    randomState: 42,
    nInit: 50
  });
  
  const labels2 = await sc2.fitPredict(X);
  const ari2 = calculateARI(y_true, labels2 as number[]);
  console.log(`   ARI: ${ari2.toFixed(4)}`);
  
  // Test 3: Multiple runs with different random states
  console.log("\n3. Multiple runs with different random states:");
  const aris = [];
  for (let seed = 0; seed < 10; seed++) {
    const sc = new SpectralClustering({
      nClusters: 2,
      affinity: "rbf",
      gamma: 1.0,
      randomState: seed
    });
    
    const labels = await sc.fitPredict(X);
    const ari = calculateARI(y_true, labels as number[]);
    aris.push(ari);
    console.log(`   Seed ${seed}: ARI = ${ari.toFixed(4)}`);
  }
  
  console.log(`\n   Mean ARI: ${(aris.reduce((a, b) => a + b) / aris.length).toFixed(4)}`);
  console.log(`   Best ARI: ${Math.max(...aris).toFixed(4)}`);
  console.log(`   Worst ARI: ${Math.min(...aris).toFixed(4)}`);
  
  // Cleanup
  X.dispose();
}

function calculateARI(labels_true: number[], labels_pred: number[]): number {
  const n = labels_true.length;
  
  // Create contingency matrix
  const classes = new Set(labels_true);
  const clusters = new Set(labels_pred);
  const contingency: Map<string, number> = new Map();
  
  for (let i = 0; i < n; i++) {
    const key = `${labels_true[i]},${labels_pred[i]}`;
    contingency.set(key, (contingency.get(key) || 0) + 1);
  }
  
  // Calculate sums
  const sum_classes = new Map<number, number>();
  const sum_clusters = new Map<number, number>();
  
  for (const cls of classes) {
    sum_classes.set(cls, labels_true.filter(l => l === cls).length);
  }
  for (const cls of clusters) {
    sum_clusters.set(cls, labels_pred.filter(l => l === cls).length);
  }
  
  // Calculate ARI components
  let sum_nij_2 = 0;
  for (const count of contingency.values()) {
    sum_nij_2 += count * (count - 1) / 2;
  }
  
  let sum_ai_2 = 0;
  for (const count of sum_classes.values()) {
    sum_ai_2 += count * (count - 1) / 2;
  }
  
  let sum_bj_2 = 0;
  for (const count of sum_clusters.values()) {
    sum_bj_2 += count * (count - 1) / 2;
  }
  
  const expected_index = sum_ai_2 * sum_bj_2 / (n * (n - 1) / 2);
  const max_index = (sum_ai_2 + sum_bj_2) / 2;
  
  if (max_index - expected_index === 0) return 1.0;
  
  return (sum_nij_2 - expected_index) / (max_index - expected_index);
}

testKMeansImprovement().catch(console.error);