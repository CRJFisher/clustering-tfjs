import * as tf from '@tensorflow/tfjs-node';
import { SpectralClustering } from './dist/clustering/spectral';
import { compute_knn_affinity } from './dist/utils/affinity';
import { normalised_laplacian } from './dist/utils/laplacian';
import { smallest_eigenvectors_with_values } from './dist/utils/smallest_eigenvectors_with_values';
import * as fs from 'fs';
import * as path from 'path';

async function debugDiffusionImplementation() {
  console.log('=== Debugging our diffusion scaling implementation ===\n');
  
  // Test with moons_n2_knn
  const fixture = JSON.parse(
    fs.readFileSync(path.join(__dirname, 'test/fixtures/spectral/moons_n2_knn.json'), 'utf-8')
  );
  
  const X = tf.tensor2d(fixture.X);
  const affinity = compute_knn_affinity(X, fixture.params.nNeighbors, true);
  
  // Get Laplacian
  const laplacian = normalised_laplacian(affinity);
  
  // Get eigenvectors and eigenvalues
  const { eigenvectors, eigenvalues } = smallest_eigenvectors_with_values(laplacian, 2);
  
  console.log('Eigenvalues:');
  const eigenvals = await eigenvalues.array();
  console.log(eigenvals.slice(0, 5));
  
  // Check our diffusion scaling
  console.log('\nDiffusion scaling factors for first 2 eigenvalues:');
  for (let i = 0; i < 2; i++) {
    const factor = Math.sqrt(Math.max(0, 1 - eigenvals[i]));
    console.log(`  eigenvalue[${i}] = ${eigenvals[i].toFixed(6)}, scaling = ${factor.toFixed(6)}`);
  }
  
  // Get first two eigenvectors
  const eigenvectorsArray = await eigenvectors.array();
  const firstEigenvector = eigenvectorsArray.map(row => row[0]);
  const secondEigenvector = eigenvectorsArray.map(row => row[1]);
  
  // Check unique values
  console.log('\nUnique values in eigenvectors:');
  console.log(`  First eigenvector: ${new Set(firstEigenvector.map(v => v.toFixed(6))).size}`);
  console.log(`  Second eigenvector: ${new Set(secondEigenvector.map(v => v.toFixed(6))).size}`);
  
  // Check if first eigenvector is constant
  const firstStd = Math.sqrt(firstEigenvector.reduce((sum, v) => {
    const mean = firstEigenvector.reduce((a, b) => a + b) / firstEigenvector.length;
    return sum + Math.pow(v - mean, 2);
  }, 0) / firstEigenvector.length);
  
  console.log(`\nFirst eigenvector std: ${firstStd.toFixed(10)}`);
  console.log(`Is constant: ${firstStd < 1e-10}`);
  
  // Apply diffusion scaling manually
  const scaledFirst = firstEigenvector.map(v => v * Math.sqrt(Math.max(0, 1 - eigenvals[0])));
  const scaledSecond = secondEigenvector.map(v => v * Math.sqrt(Math.max(0, 1 - eigenvals[1])));
  
  console.log('\nAfter diffusion scaling:');
  console.log(`  First eigenvector unique values: ${new Set(scaledFirst.map(v => v.toFixed(6))).size}`);
  console.log(`  Second eigenvector unique values: ${new Set(scaledSecond.map(v => v.toFixed(6))).size}`);
  
  // Create embedding matrix
  const embedding = tf.stack([
    tf.tensor1d(scaledFirst),
    tf.tensor1d(scaledSecond)
  ], 1) as tf.Tensor2D;
  
  console.log('\nEmbedding shape:', embedding.shape);
  
  // Show first few rows
  const embeddingArray = await embedding.array();
  console.log('\nFirst 5 rows of embedding:');
  embeddingArray.slice(0, 5).forEach((row, i) => {
    console.log(`  [${i}]: [${row[0].toFixed(6)}, ${row[1].toFixed(6)}]`);
  });
  
  // Clean up
  X.dispose();
  affinity.dispose();
  laplacian.dispose();
  eigenvectors.dispose();
  eigenvalues.dispose();
  embedding.dispose();
}

debugDiffusionImplementation().catch(console.error);