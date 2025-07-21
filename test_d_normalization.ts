import * as tf from "@tensorflow/tfjs-node";
import { SpectralClustering } from "./src/clustering/spectral";
import { normalised_laplacian } from "./src/utils/laplacian";
import { compute_rbf_affinity } from "./src/utils/affinity";
import { smallest_eigenvectors_with_values } from "./src/utils/smallest_eigenvectors_with_values";
import * as fs from "fs";

async function testDNormalization() {
  // Load fixture and sklearn results
  const fixture = JSON.parse(
    fs.readFileSync("./test/fixtures/spectral/circles_n2_rbf.json", "utf8")
  );
  const sklearnResults = JSON.parse(
    fs.readFileSync("./sklearn_workflow_debug.json", "utf8")
  );
  
  console.log("Testing D^{-1/2} normalization");
  console.log("=" .repeat(60) + "\n");
  
  const X = tf.tensor2d(fixture.X);
  const nClusters = fixture.params.nClusters;
  
  // Compute affinity and Laplacian with degree info
  const affinity = compute_rbf_affinity(X, fixture.params.gamma);
  const { laplacian, sqrtDegrees } = normalised_laplacian(affinity, true);
  
  // Get eigenvectors
  const { eigenvectors: U_full, eigenvalues } = smallest_eigenvectors_with_values(
    laplacian, 
    nClusters
  );
  
  const eigenvecs = await U_full.array();
  const eigenvals = await eigenvalues.array();
  const sqrtDeg = await sqrtDegrees.array();
  
  console.log("1. Our raw eigenvectors (first row):");
  console.log(`   ${eigenvecs[0].map(v => v.toFixed(6)).join(", ")}`);
  
  console.log("\n2. Sqrt degrees (D^{-1/2}) first 5 elements:");
  console.log(`   ${sqrtDeg.slice(0, 5).map(v => v.toFixed(6)).join(", ")}`);
  
  console.log("\n3. Testing different normalizations:");
  
  // Method 1: Just diffusion map scaling
  const scaling1 = Math.sqrt(Math.max(0, 1 - eigenvals[1]));
  const embed1_1 = eigenvecs[0][1] * scaling1;
  console.log(`   Eigenvector * sqrt(1-lambda): ${embed1_1.toFixed(6)}`);
  
  // Method 2: Divide by sqrt degree (what we just implemented)
  const embed2_1 = embed1_1 / sqrtDeg[0];
  console.log(`   Above / sqrtDegree: ${embed2_1.toFixed(6)}`);
  
  // Method 3: Multiply by sqrt degree (inverse of what sklearn comment says)
  const embed3_1 = embed1_1 * sqrtDeg[0];
  console.log(`   Eigenvector * sqrt(1-lambda) * sqrtDegree: ${embed3_1.toFixed(6)}`);
  
  console.log(`\n4. Sklearn's embedding (first element of second eigenvector): ${sklearnResults.embedding[0][1].toFixed(6)}`);
  
  // Check which matches
  const diff1 = Math.abs(embed1_1 - sklearnResults.embedding[0][1]);
  const diff2 = Math.abs(embed2_1 - sklearnResults.embedding[0][1]);
  const diff3 = Math.abs(embed3_1 - sklearnResults.embedding[0][1]);
  
  console.log("\n5. Differences from sklearn:");
  console.log(`   Method 1 (no D normalization): ${diff1.toFixed(6)}`);
  console.log(`   Method 2 (divide by sqrtDegree): ${diff2.toFixed(6)}`);
  console.log(`   Method 3 (multiply by sqrtDegree): ${diff3.toFixed(6)}`);
  
  // Cleanup
  X.dispose();
  affinity.dispose();
  laplacian.dispose();
  sqrtDegrees.dispose();
  U_full.dispose();
  eigenvalues.dispose();
}

testDNormalization().catch(console.error);