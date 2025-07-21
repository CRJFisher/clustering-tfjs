import { SpectralClustering } from "./src/clustering/spectral";
import { compute_rbf_affinity } from "./src/utils/affinity";
import * as tf from "@tensorflow/tfjs-node";
import * as fs from "fs";

async function testFix() {
  // Load fixture
  const fixture = JSON.parse(fs.readFileSync("./test/fixtures/spectral/circles_n2_rbf.json", "utf8"));
  const X = tf.tensor2d(fixture.X);
  const y_true = fixture.labels;
  
  // Create affinity
  const affinity = compute_rbf_affinity(X, 1.0);
  
  // Run spectral clustering
  const sc = new SpectralClustering({
    nClusters: 2,
    affinity: "precomputed",
    randomState: 42
  });
  
  const labels = await sc.fitPredict(affinity);
  
  // Calculate ARI manually
  function adjustedRandIndex(labels_true: number[], labels_pred: number[]): number {
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
  
  const labelsArray = Array.isArray(labels) ? labels : await (labels as any).array();
  const ari = adjustedRandIndex(y_true, labelsArray);
  
  console.log(`ARI: ${ari}`);
  console.log(`Expected: ~1.0`);
  
  // Cleanup
  X.dispose();
  affinity.dispose();
}

testFix().catch(console.error);