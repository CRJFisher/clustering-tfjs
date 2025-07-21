import * as tf from "@tensorflow/tfjs-node";
import { compute_rbf_affinity } from "./src/utils/affinity";
import { createConstantEigenvector } from "./src/utils/constant_eigenvector";
import { normalised_laplacian } from "./src/utils/laplacian";
import { smallest_eigenvectors_with_values } from "./src/utils/smallest_eigenvectors_with_values";
import * as fs from "fs";

async function debug() {
  // Load fixture
  const fixture = JSON.parse(fs.readFileSync("./test/fixtures/spectral/circles_n2_rbf.json", "utf8"));
  const X = tf.tensor2d(fixture.X);
  
  // Create affinity
  const affinity = compute_rbf_affinity(X, 1.0);
  
  // Get Laplacian
  const L = normalised_laplacian(affinity);
  
  // Get eigenvectors
  const { eigenvectors, eigenvalues } = smallest_eigenvectors_with_values(L, 2);
  
  console.log("Eigenvalues:", await eigenvalues.array());
  
  // Get constant eigenvector
  const constantEigenvec = createConstantEigenvector(affinity);
  
  console.log("\nConstant eigenvector:");
  const constArray = await constantEigenvec.array() as number[][];
  console.log("Shape:", constantEigenvec.shape);
  console.log("First 5 values:", constArray.slice(0, 5).map(row => row[0]));
  console.log("All same?", constArray.every(row => Math.abs(row[0] - constArray[0][0]) < 1e-10));
  
  // Check scaling
  const scaled = tf.mul(constantEigenvec, tf.sqrt(tf.sub(1, 0))); // sqrt(1 - 0) = 1
  console.log("\nScaled constant:");
  const scaledArray = await scaled.array() as number[][];
  console.log("First value:", scaledArray[0][0]);
  console.log("L2 norm:", tf.norm(scaled).arraySync());
  
  // Compare with sklearn's value
  console.log("\nsklearn's constant value: 0.0264185062");
  console.log("Our value:", scaledArray[0][0]);
  console.log("Ratio:", 0.0264185062 / scaledArray[0][0]);
  
  // Cleanup
  X.dispose();
  affinity.dispose();
  L.dispose();
  eigenvectors.dispose();
  eigenvalues.dispose();
  constantEigenvec.dispose();
  scaled.dispose();
}

debug().catch(console.error);