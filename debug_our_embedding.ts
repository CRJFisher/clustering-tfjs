import { SpectralClustering } from "./src/clustering/spectral";
import { compute_rbf_affinity } from "./src/utils/affinity";
import { normalised_laplacian } from "./src/utils/laplacian";
import { smallest_eigenvectors_with_values } from "./src/utils/smallest_eigenvectors_with_values";
import { createConstantEigenvector } from "./src/utils/constant_eigenvector";
import * as tf from "@tensorflow/tfjs-node";
import * as fs from "fs";

async function debugEmbedding() {
  // Load fixture
  const fixture = JSON.parse(fs.readFileSync("./test/fixtures/spectral/circles_n2_rbf.json", "utf8"));
  const X = tf.tensor2d(fixture.X);
  
  // Create affinity
  const affinity = compute_rbf_affinity(X, 1.0);
  
  // Get Laplacian and eigenvectors
  const L = normalised_laplacian(affinity);
  const { eigenvectors, eigenvalues } = smallest_eigenvectors_with_values(L, 3);
  
  console.log("Eigenvalues:", await eigenvalues.array());
  
  // Create embedding manually
  const constantEigenvec = createConstantEigenvector(affinity);
  const secondEigenvec = tf.slice(eigenvectors, [0, 1], [-1, 1]);
  
  // Scale them
  const const_scaled = tf.mul(constantEigenvec, tf.sqrt(tf.sub(1, 0)));
  const eigval1 = (await eigenvalues.array())[1];
  const second_scaled = tf.mul(secondEigenvec, tf.sqrt(tf.sub(1, eigval1)));
  
  const embedding = tf.concat([const_scaled, second_scaled], 1);
  
  console.log("\nOur embedding:");
  const embArray = await embedding.array() as number[][];
  console.log("Shape:", embedding.shape);
  console.log("Dim 0 first value:", embArray[0][0]);
  console.log("Dim 0 all same?", embArray.every(row => Math.abs(row[0] - embArray[0][0]) < 1e-10));
  console.log("Dim 1 range:", [
    Math.min(...embArray.map(row => row[1])),
    Math.max(...embArray.map(row => row[1]))
  ]);
  
  // Run k-means on just dimension 1
  const { KMeans } = require("./src/clustering/kmeans");
  const km = new KMeans({ nClusters: 2, randomState: 42, nInit: 10 });
  const dim1Only = tf.slice(embedding, [0, 1], [-1, 1]);
  await km.fit(dim1Only);
  
  console.log("\nK-means on dimension 1 only:");
  console.log("Labels:", km.labels_);
  
  // Calculate ARI
  function ari(labels_true: number[], labels_pred: number[]): number {
    // ... simplified ARI calculation ...
    const contingency: Map<string, number> = new Map();
    for (let i = 0; i < labels_true.length; i++) {
      const key = `${labels_true[i]},${labels_pred[i]}`;
      contingency.set(key, (contingency.get(key) || 0) + 1);
    }
    
    // If perfect match
    if (contingency.size === 2 && 
        contingency.get("0,0") === 30 && contingency.get("1,1") === 30) return 1.0;
    if (contingency.size === 2 && 
        contingency.get("0,1") === 30 && contingency.get("1,0") === 30) return 1.0;
    
    return -1; // Not perfect
  }
  
  console.log("ARI:", ari(fixture.labels, km.labels_ as number[]));
  
  // Cleanup
  X.dispose();
  affinity.dispose();
  L.dispose();
  eigenvectors.dispose();
  eigenvalues.dispose();
  constantEigenvec.dispose();
  secondEigenvec.dispose();
  const_scaled.dispose();
  second_scaled.dispose();
  embedding.dispose();
  dim1Only.dispose();
}

debugEmbedding().catch(console.error);