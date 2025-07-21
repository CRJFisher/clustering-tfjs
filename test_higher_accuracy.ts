import * as tf from "@tensorflow/tfjs-node";
import { improved_jacobi_eigen } from "./src/utils/eigen_improved";
import { normalised_laplacian } from "./src/utils/laplacian";
import { compute_rbf_affinity } from "./src/utils/affinity";
import * as fs from "fs";

async function testAccuracy() {
  // Load fixture
  const fixture = JSON.parse(fs.readFileSync("./test/fixtures/spectral/circles_n2_rbf.json", "utf8"));
  const X = tf.tensor2d(fixture.X);
  
  // Create affinity and Laplacian
  const affinity = compute_rbf_affinity(X, 1.0);
  const L = normalised_laplacian(affinity);
  const Larray = await L.array();
  
  console.log("Testing different eigendecomposition settings:\n");
  
  // Test 1: Current settings
  const result1 = improved_jacobi_eigen(tf.tensor2d(Larray), {
    isPSD: true,
    maxIterations: 3000,
    tolerance: 1e-14,
  });
  console.log("1. Current (3000 iter, 1e-14 tol):");
  console.log(`   Eigenvalue 1: ${result1.eigenvalues[1]}`);
  console.log(`   Eigenvector 1 first element: ${result1.eigenvectors[0][1]}`);
  
  // Test 2: More iterations
  const result2 = improved_jacobi_eigen(tf.tensor2d(Larray), {
    isPSD: true,
    maxIterations: 10000,
    tolerance: 1e-14,
  });
  console.log("\n2. More iterations (10000 iter, 1e-14 tol):");
  console.log(`   Eigenvalue 1: ${result2.eigenvalues[1]}`);
  console.log(`   Eigenvector 1 first element: ${result2.eigenvectors[0][1]}`);
  console.log(`   Difference: ${Math.abs(result2.eigenvectors[0][1] - result1.eigenvectors[0][1])}`);
  
  // Test 3: Tighter tolerance
  const result3 = improved_jacobi_eigen(tf.tensor2d(Larray), {
    isPSD: true,
    maxIterations: 10000,
    tolerance: 1e-16,
  });
  console.log("\n3. Tighter tolerance (10000 iter, 1e-16 tol):");
  console.log(`   Eigenvalue 1: ${result3.eigenvalues[1]}`);
  console.log(`   Eigenvector 1 first element: ${result3.eigenvectors[0][1]}`);
  console.log(`   Difference from (1): ${Math.abs(result3.eigenvectors[0][1] - result1.eigenvectors[0][1])}`);
  
  // Compare with expected sklearn value
  // From our analysis, sklearn's eigenvector[1][0] = 0.16337048
  const sklearn_val = 0.16337048;
  console.log("\n4. Comparison with sklearn:");
  console.log(`   sklearn eigenvector[1][0]: ${sklearn_val}`);
  console.log(`   Our best: ${result3.eigenvectors[0][1]}`);
  console.log(`   Difference: ${Math.abs(result3.eigenvectors[0][1] - sklearn_val)}`);
  console.log(`   Relative error: ${(Math.abs(result3.eigenvectors[0][1] - sklearn_val) / Math.abs(sklearn_val) * 100).toFixed(2)}%`);
  
  // Cleanup
  X.dispose();
  affinity.dispose();
  L.dispose();
}

testAccuracy().catch(console.error);