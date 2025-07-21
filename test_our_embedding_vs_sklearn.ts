import * as tf from "@tensorflow/tfjs-node";
import { SpectralClustering } from "./src/clustering/spectral";
import * as fs from "fs";

async function compareEmbeddings() {
  // Load fixture and sklearn results
  const fixture = JSON.parse(
    fs.readFileSync("./test/fixtures/spectral/circles_n2_rbf.json", "utf8")
  );
  const sklearnResults = JSON.parse(
    fs.readFileSync("./sklearn_workflow_debug.json", "utf8")
  );
  
  console.log("Comparing our embedding computation with sklearn's");
  console.log("=" .repeat(60) + "\n");
  
  const X = tf.tensor2d(fixture.X);
  
  // Create SpectralClustering instance to trace through
  const sc = new SpectralClustering({
    nClusters: fixture.params.nClusters,
    affinity: "rbf",
    gamma: fixture.params.gamma,
    randomState: fixture.params.randomState
  });
  
  // Manually run through the steps to see where we diverge
  console.log("1. Computing affinity matrix...");
  const { compute_rbf_affinity } = await import("./src/utils/affinity");
  const affinity = compute_rbf_affinity(X, fixture.params.gamma);
  
  console.log("2. Computing normalized Laplacian...");
  const { normalised_laplacian } = await import("./src/utils/laplacian");
  const laplacian = normalised_laplacian(affinity);
  
  console.log("3. Computing eigenvectors...");
  const { smallest_eigenvectors_with_values } = await import("./src/utils/smallest_eigenvectors_with_values");
  const { eigenvectors: U_full, eigenvalues } = smallest_eigenvectors_with_values(
    laplacian, 
    fixture.params.nClusters
  );
  
  console.log("\n4. Our current embedding computation:");
  const U_scaled = tf.tidy(() => {
    const eigenvals = tf.slice(eigenvalues, [0], [fixture.params.nClusters]) as tf.Tensor1D;
    const scalingFactors = tf.sqrt(
      tf.maximum(tf.scalar(0), tf.sub(tf.scalar(1), eigenvals))
    ) as tf.Tensor1D;
    const scalingFactors2D = scalingFactors.reshape([1, -1]) as tf.Tensor2D;
    const U_selected = tf.slice(U_full, [0, 0], [-1, fixture.params.nClusters]) as tf.Tensor2D;
    return U_selected.mul(scalingFactors2D) as tf.Tensor2D;
  });
  
  const ourEmbedding = await U_scaled.array();
  const sklearnEmbedding = sklearnResults.embedding;
  
  console.log(`   Our embedding shape: [${U_scaled.shape}]`);
  console.log(`   Our first row: [${ourEmbedding[0].map(v => v.toFixed(6)).join(", ")}]`);
  console.log(`   Sklearn first row: [${sklearnEmbedding[0].map((v: number) => v.toFixed(6)).join(", ")}]`);
  
  // Check element-wise differences
  let maxDiff = 0;
  for (let i = 0; i < ourEmbedding.length; i++) {
    for (let j = 0; j < ourEmbedding[i].length; j++) {
      const diff = Math.abs(ourEmbedding[i][j] - sklearnEmbedding[i][j]);
      maxDiff = Math.max(maxDiff, diff);
    }
  }
  console.log(`   Max difference: ${maxDiff.toFixed(6)}`);
  
  // Check if there's a constant scaling factor
  const ratio0 = sklearnEmbedding[0][0] / ourEmbedding[0][0];
  const ratio1 = sklearnEmbedding[0][1] / ourEmbedding[0][1];
  console.log(`   Ratio sklearn/ours for first element: ${ratio0.toFixed(6)}`);
  console.log(`   Ratio sklearn/ours for second element: ${ratio1.toFixed(6)}`);
  
  // Try running k-means on both embeddings
  console.log("\n5. K-means results:");
  const { KMeans } = await import("./src/clustering/kmeans");
  
  // On our embedding
  const km1 = new KMeans({
    nClusters: fixture.params.nClusters,
    randomState: fixture.params.randomState,
    nInit: 10
  });
  await km1.fit(U_scaled);
  const labels1 = km1.labels_ as number[];
  const ari1 = calculateARI(fixture.labels, labels1);
  console.log(`   ARI with our embedding: ${ari1.toFixed(4)}`);
  
  // On sklearn embedding
  const sklearnTensor = tf.tensor2d(sklearnEmbedding);
  const km2 = new KMeans({
    nClusters: fixture.params.nClusters,
    randomState: fixture.params.randomState,
    nInit: 10
  });
  await km2.fit(sklearnTensor);
  const labels2 = km2.labels_ as number[];
  const ari2 = calculateARI(fixture.labels, labels2);
  console.log(`   ARI with sklearn embedding: ${ari2.toFixed(4)}`);
  
  // Cleanup
  X.dispose();
  affinity.dispose();
  laplacian.dispose();
  U_full.dispose();
  eigenvalues.dispose();
  U_scaled.dispose();
  sklearnTensor.dispose();
}

function calculateARI(labels_true: number[], labels_pred: number[]): number {
  const n = labels_true.length;
  
  const classes = new Set(labels_true);
  const clusters = new Set(labels_pred);
  const contingency: Map<string, number> = new Map();
  
  for (let i = 0; i < n; i++) {
    const key = `${labels_true[i]},${labels_pred[i]}`;
    contingency.set(key, (contingency.get(key) || 0) + 1);
  }
  
  const sum_classes = new Map<number, number>();
  const sum_clusters = new Map<number, number>();
  
  for (const cls of classes) {
    sum_classes.set(cls, labels_true.filter(l => l === cls).length);
  }
  for (const cls of clusters) {
    sum_clusters.set(cls, labels_pred.filter(l => l === cls).length);
  }
  
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

compareEmbeddings().catch(console.error);