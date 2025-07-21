
import * as tf from "@tensorflow/tfjs-node";
import { improved_jacobi_eigen } from "./src/utils/eigen_improved";
import { normalised_laplacian } from "./src/utils/laplacian";
import { compute_rbf_affinity } from "./src/utils/affinity";
import { smallest_eigenvectors_with_values } from "./src/utils/smallest_eigenvectors_with_values";
import * as fs from "fs";

async function main() {
  const fixture = JSON.parse(fs.readFileSync("./test/fixtures/spectral/circles_n2_rbf.json", "utf8"));
  const X = tf.tensor2d(fixture.X);
  
  // Create affinity
  const affinity = compute_rbf_affinity(X, 1.0);
  
  // Get Laplacian
  const L = normalised_laplacian(affinity);
  
  // Get our eigenvectors
  const { eigenvectors, eigenvalues } = smallest_eigenvectors_with_values(L, 2);
  
  // Also run full decomposition for comparison
  const Larray = await L.array();
  const { eigenvalues: allEvals, eigenvectors: allEvecs } = improved_jacobi_eigen(
    tf.tensor2d(Larray),
    { isPSD: true, maxIterations: 3000, tolerance: 1e-14 }
  );
  
  // Save results
  const results = {
    selectedEigenvalues: await eigenvalues.array(),
    selectedEigenvectors: await eigenvectors.array(),
    allEigenvalues: allEvals,
    firstThreeEigenvectors: allEvecs.map(row => row.slice(0, 3))
  };
  
  fs.writeFileSync("our_eigen_results.json", JSON.stringify(results, null, 2));
  console.log("Saved eigenvector results to our_eigen_results.json");
  
  // Cleanup
  X.dispose();
  affinity.dispose();
  L.dispose();
  eigenvectors.dispose();
  eigenvalues.dispose();
}

main().catch(console.error);
