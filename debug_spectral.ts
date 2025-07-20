import * as tf from "@tensorflow/tfjs-node";
import { SpectralClustering } from "./src/clustering/spectral";
import * as fs from "fs";
import * as path from "path";

// Debug spectral clustering implementation
async function debug() {
  // Load a failing test case
  const data = JSON.parse(
    fs.readFileSync(
      path.join(__dirname, "test/fixtures/spectral/circles_n2_rbf.json"),
      "utf-8"
    )
  );

  const X = tf.tensor2d(data.X as number[][]);
  console.log("Input shape:", X.shape);
  
  const model = new SpectralClustering({
    nClusters: data.params.nClusters,
    affinity: data.params.affinity,
    gamma: data.params.gamma,
    randomState: data.params.randomState,
  });

  // Monkey patch to inspect intermediate values
  const originalFit = model.fit.bind(model);
  model.fit = async function(X) {
    console.log("\n=== Starting fit ===");
    
    // Hook into the affinity matrix computation
    const originalCompute = (SpectralClustering as any).computeAffinityMatrix;
    (SpectralClustering as any).computeAffinityMatrix = function(...args: any[]) {
      const result = originalCompute.apply(this, args);
      console.log("Affinity matrix shape:", result.shape);
      console.log("Affinity matrix min:", result.min().dataSync()[0]);
      console.log("Affinity matrix max:", result.max().dataSync()[0]);
      console.log("Affinity matrix sum:", result.sum().dataSync()[0]);
      return result;
    };

    await originalFit.call(this, X);
  };

  await model.fit(X);
  
  console.log("\nOur labels:", model.labels_);
  console.log("Expected labels:", data.labels);
  
  // Check label distribution
  const ourCounts: Record<number, number> = {};
  const expectedCounts: Record<number, number> = {};
  
  for (const label of model.labels_!) {
    ourCounts[label] = (ourCounts[label] || 0) + 1;
  }
  
  if (data.labels) {
    for (const label of data.labels) {
      expectedCounts[label] = (expectedCounts[label] || 0) + 1;
    }
  }
  
  console.log("\nOur label counts:", ourCounts);
  console.log("Expected label counts:", expectedCounts);
  
  X.dispose();
}

debug().catch(console.error);