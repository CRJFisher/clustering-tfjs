import * as tf from "@tensorflow/tfjs-node";
import { compute_rbf_affinity } from "./src/utils/affinity";
import { normalised_laplacian, jacobi_eigen_decomposition, degree_vector } from "./src/utils/laplacian";
import * as fs from "fs";
import * as path from "path";

// Debug our eigenvalue computation
async function debug() {
  // Load a failing test case
  const data = JSON.parse(
    fs.readFileSync(
      path.join(__dirname, "test/fixtures/spectral/circles_n2_rbf.json"),
      "utf-8"
    )
  );

  const X = tf.tensor2d(data.X as number[][]);
  
  // Step 1: Compute affinity matrix
  const affinity = compute_rbf_affinity(X, data.params.gamma);
  
  // Step 2: Compute Laplacian
  const laplacian = normalised_laplacian(affinity);
  const lapArr = await laplacian.array();
  
  console.log("Our Laplacian:");
  console.log("Diagonal (first 5):", [0,1,2,3,4].map(i => lapArr[i][i]));
  console.log("First row (first 5 non-diag):", lapArr[0].slice(1, 6));
  
  // Step 3: Get eigendecomposition
  const { eigenvalues, eigenvectors } = jacobi_eigen_decomposition(laplacian);
  
  console.log("Eigenvalues (first 10):");
  console.log(eigenvalues.slice(0, 10));
  
  console.log("\nSmallest eigenvalue:", eigenvalues[0]);
  console.log("Second smallest eigenvalue:", eigenvalues[1]);
  
  console.log("\nFirst eigenvector (first 5 values):");
  console.log(eigenvectors.slice(0, 5).map(row => row[0]));
  
  // Compute statistics
  const firstCol = eigenvectors.map(row => row[0]);
  const min = Math.min(...firstCol);
  const max = Math.max(...firstCol);
  const mean = firstCol.reduce((a, b) => a + b, 0) / firstCol.length;
  const std = Math.sqrt(firstCol.reduce((a, b) => a + (b - mean) ** 2, 0) / firstCol.length);
  
  console.log("\nFirst eigenvector statistics:");
  console.log(`  min: ${min}`);
  console.log(`  max: ${max}`);
  console.log(`  std: ${std}`);
  
  // Apply D^{-1/2} normalization
  const deg = degree_vector(affinity);
  const degArr = await deg.array();
  
  const normalized = eigenvectors.map((row, i) => {
    const d = degArr[i];
    return row.map(val => val / Math.sqrt(d));
  });
  
  console.log("\nAfter D^{-1/2} normalization:");
  console.log("First column (first 5 values):", normalized.slice(0, 5).map(row => row[0]));
  
  const firstColNorm = normalized.map(row => row[0]);
  const minNorm = Math.min(...firstColNorm);
  const maxNorm = Math.max(...firstColNorm);
  const meanNorm = firstColNorm.reduce((a, b) => a + b, 0) / firstColNorm.length;
  const stdNorm = Math.sqrt(firstColNorm.reduce((a, b) => a + (b - meanNorm) ** 2, 0) / firstColNorm.length);
  
  console.log("First column statistics:");
  console.log(`  min: ${minNorm}`);
  console.log(`  max: ${maxNorm}`);
  console.log(`  std: ${stdNorm}`);
  
  // Clean up
  X.dispose();
  affinity.dispose();
  laplacian.dispose();
  deg.dispose();
}

debug().catch(console.error);