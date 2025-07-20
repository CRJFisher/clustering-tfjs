import * as tf from "@tensorflow/tfjs-node";
import { SpectralClustering } from "./src/clustering/spectral";
import { compute_rbf_affinity } from "./src/utils/affinity";
import { normalised_laplacian, smallest_eigenvectors, degree_vector } from "./src/utils/laplacian";
import * as fs from "fs";
import * as path from "path";

// Debug spectral clustering implementation with detailed intermediate outputs
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
  console.log("Params:", data.params);
  
  // Step 1: Compute affinity matrix
  const affinity = compute_rbf_affinity(X, data.params.gamma);
  console.log("\n=== Affinity Matrix ===");
  console.log("Shape:", affinity.shape);
  console.log("Min:", (await affinity.min().data())[0]);
  console.log("Max:", (await affinity.max().data())[0]);
  console.log("Sum:", (await affinity.sum().data())[0]);
  
  // Check a few values
  const affinityArr = await affinity.array();
  console.log("First row (first 5 values):", affinityArr[0].slice(0, 5));
  console.log("Diagonal (first 5):", [0,1,2,3,4].map(i => affinityArr[i][i]));
  
  // Step 2: Compute degree vector
  const deg = degree_vector(affinity);
  console.log("\n=== Degree Vector ===");
  const degArr = await deg.array();
  console.log("First 5 values:", degArr.slice(0, 5));
  console.log("Min degree:", Math.min(...degArr));
  console.log("Max degree:", Math.max(...degArr));
  
  // Step 3: Compute Laplacian
  const laplacian = normalised_laplacian(affinity);
  console.log("\n=== Normalized Laplacian ===");
  console.log("Shape:", laplacian.shape);
  const lapArr = await laplacian.array();
  console.log("Diagonal (first 5):", [0,1,2,3,4].map(i => lapArr[i][i]));
  console.log("First row (first 5 values):", lapArr[0].slice(0, 5));
  
  // Step 4: Get eigenvectors
  const U_raw = smallest_eigenvectors(laplacian, data.params.nClusters);
  console.log("\n=== Raw Eigenvectors ===");
  console.log("Shape:", U_raw.shape);
  const U_rawArr = await U_raw.array();
  console.log("First 5 rows:");
  for (let i = 0; i < 5; i++) {
    console.log(`  Row ${i}:`, U_rawArr[i]);
  }
  
  // Step 5: Apply D^{-1/2} normalization
  const invSqrtDeg = tf.where(
    deg.equal(0),
    tf.zerosLike(deg),
    deg.pow(-0.5),
  ) as tf.Tensor1D;
  const U_norm = U_raw.div(invSqrtDeg.expandDims(1)) as tf.Tensor2D;
  
  console.log("\n=== Normalized Eigenvectors (after D^{-1/2}) ===");
  const U_normArr = await U_norm.array();
  console.log("First 5 rows:");
  for (let i = 0; i < 5; i++) {
    console.log(`  Row ${i}:`, U_normArr[i]);
  }
  
  // Check for constant columns
  console.log("\n=== Checking for constant eigenvectors ===");
  const nCols = U_normArr[0].length;
  for (let col = 0; col < nCols; col++) {
    const colValues = U_normArr.map(row => row[col]);
    const min = Math.min(...colValues);
    const max = Math.max(...colValues);
    const mean = colValues.reduce((a, b) => a + b, 0) / colValues.length;
    const std = Math.sqrt(colValues.reduce((a, b) => a + (b - mean) ** 2, 0) / colValues.length);
    console.log(`Column ${col}: min=${min.toFixed(6)}, max=${max.toFixed(6)}, std=${std.toFixed(6)}`);
  }
  
  // Also check what the first nClusters columns look like
  console.log("\n=== First nClusters columns after slicing ===");
  const U_sliced = U_normArr.map(row => row.slice(0, data.params.nClusters));
  console.log("First 5 rows:");
  for (let i = 0; i < 5; i++) {
    console.log(`  Row ${i}:`, U_sliced[i]);
  }
  
  // Clean up
  X.dispose();
  affinity.dispose();
  deg.dispose();
  laplacian.dispose();
  U_raw.dispose();
  invSqrtDeg.dispose();
  U_norm.dispose();
}

debug().catch(console.error);