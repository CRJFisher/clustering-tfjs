import * as tf from "@tensorflow/tfjs-node";
import { Matrix, EigenvalueDecomposition } from "ml-matrix";
import { normalised_laplacian } from "./src/utils/laplacian";
import { compute_rbf_affinity } from "./src/utils/affinity";
import * as fs from "fs";

async function compareWithSklearn() {
  // Load fixture
  const fixture = JSON.parse(
    fs.readFileSync("./test/fixtures/spectral/circles_n2_rbf.json", "utf8")
  );
  
  // Load sklearn results if available
  const sklearnPath = "./sklearn_laplacian_results.json";
  if (!fs.existsSync(sklearnPath)) {
    console.log("sklearn_laplacian_results.json not found. Please run the Python comparison script first.");
    return;
  }
  
  const sklearnResults = JSON.parse(fs.readFileSync(sklearnPath, "utf8"));
  
  console.log("Comparing our eigenvectors with sklearn's ARPACK results");
  console.log("=" .repeat(60) + "\n");
  
  const X = tf.tensor2d(fixture.X);
  const nClusters = fixture.params.nClusters;
  
  // Compute our Laplacian
  const affinity = compute_rbf_affinity(X, fixture.params.gamma);
  const laplacian = normalised_laplacian(affinity);
  const laplacianArray = await laplacian.array();
  
  // Get eigenvectors using ml-matrix
  const mlMatrix = new Matrix(laplacianArray);
  const eigen = new EigenvalueDecomposition(mlMatrix, { assumeSymmetric: true });
  
  // Sort by eigenvalue
  const mlEigenvalues = eigen.realEigenvalues;
  const mlEigenvectors = eigen.eigenvectorMatrix.to2DArray();
  const eigenPairs = mlEigenvalues.map((val, idx) => ({
    value: val,
    vector: mlEigenvectors.map(row => row[idx])
  }));
  eigenPairs.sort((a, b) => a.value - b.value);
  
  console.log("1. Eigenvalue comparison:");
  console.log("-".repeat(40));
  console.log("Our eigenvalues (first 5):", eigenPairs.slice(0, 5).map(p => p.value));
  console.log("Sklearn eigenvalues (first 5):", sklearnResults.eigenvalues.slice(0, 5));
  console.log("Max eigenvalue difference:", 
    Math.max(...sklearnResults.eigenvalues.slice(0, 5).map((val: number, i: number) => 
      Math.abs(val - eigenPairs[i].value)
    ))
  );
  
  console.log("\n2. Eigenvector comparison:");
  console.log("-".repeat(40));
  
  // Compare first few eigenvectors
  for (let i = 0; i < Math.min(3, nClusters); i++) {
    const ourVec = eigenPairs[i].vector;
    const sklearnVec = sklearnResults.eigenvectors[i];
    
    // Handle sign ambiguity - align signs based on first non-zero element
    let signIdx = 0;
    while (Math.abs(ourVec[signIdx]) < 1e-10 && signIdx < ourVec.length) signIdx++;
    const sign = Math.sign(ourVec[signIdx]) * Math.sign(sklearnVec[signIdx]);
    const sklearnVecAligned = sklearnVec.map((v: number) => v * sign);
    
    // Compute differences
    let maxDiff = 0;
    let avgDiff = 0;
    for (let j = 0; j < ourVec.length; j++) {
      const diff = Math.abs(ourVec[j] - sklearnVecAligned[j]);
      maxDiff = Math.max(maxDiff, diff);
      avgDiff += diff;
    }
    avgDiff /= ourVec.length;
    
    console.log(`\nEigenvector ${i}:`);
    console.log(`  Our first 5 elements:     [${ourVec.slice(0, 5).map(v => v.toFixed(6)).join(", ")}]`);
    console.log(`  Sklearn first 5 elements: [${sklearnVecAligned.slice(0, 5).map((v: number) => v.toFixed(6)).join(", ")}]`);
    console.log(`  Max difference: ${maxDiff.toFixed(6)}`);
    console.log(`  Avg difference: ${avgDiff.toFixed(6)}`);
  }
  
  console.log("\n3. Checking if sklearn uses special handling:");
  console.log("-".repeat(40));
  
  // Check if sklearn might be using shift-invert or other techniques
  const firstEigenvalue = eigenPairs[0].value;
  console.log(`First eigenvalue: ${firstEigenvalue}`);
  console.log(`Is it close to zero? ${Math.abs(firstEigenvalue) < 1e-6}`);
  
  // Check condition number
  const maxEigenvalue = Math.max(...mlEigenvalues);
  const minNonZeroEigenvalue = Math.min(...mlEigenvalues.filter((v: number) => Math.abs(v) > 1e-10));
  const conditionNumber = maxEigenvalue / minNonZeroEigenvalue;
  console.log(`Condition number: ${conditionNumber.toFixed(2)}`);
  console.log(`Matrix is ill-conditioned: ${conditionNumber > 1e6}`);
  
  // Cleanup
  X.dispose();
  affinity.dispose();
  laplacian.dispose();
}

compareWithSklearn().catch(console.error);