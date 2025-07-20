import * as tf from '@tensorflow/tfjs-node';
import { jacobi_eigen_decomposition } from './src/utils/laplacian';
import { improved_jacobi_eigen } from './src/utils/eigen_improved';

async function testImprovedJacobi() {
  console.log('Testing improved Jacobi eigensolver\n');
  
  // Test on actual normalized Laplacian
  const fixture = require('./test/fixtures/spectral/blobs_n3_knn.json');
  const { compute_knn_affinity } = require('./src/utils/affinity');
  const { normalised_laplacian } = require('./src/utils/laplacian');
  
  const X = tf.tensor2d(fixture.X);
  const affinity = compute_knn_affinity(X, fixture.params.nNeighbors, true);
  const laplacian = normalised_laplacian(affinity);
  
  console.log('Matrix size:', laplacian.shape);
  console.log('Testing on normalized Laplacian (known to be PSD)\n');
  
  // Original Jacobi
  console.log('Original Jacobi (2000 iterations):');
  let start = Date.now();
  const original = jacobi_eigen_decomposition(laplacian, { maxIterations: 2000 });
  let time = Date.now() - start;
  console.log(`Time: ${time}ms`);
  console.log('First 5 eigenvalues:', original.eigenvalues.slice(0, 5).map(v => v.toExponential(3)));
  console.log('Smallest eigenvalue:', Math.min(...original.eigenvalues).toExponential(3));
  
  // Improved Jacobi
  console.log('\nImproved Jacobi:');
  start = Date.now();
  const improved = improved_jacobi_eigen(laplacian, { isPSD: true });
  time = Date.now() - start;
  console.log(`Time: ${time}ms`);
  console.log('First 5 eigenvalues:', improved.eigenvalues.slice(0, 5).map(v => v.toExponential(3)));
  console.log('Smallest eigenvalue:', Math.min(...improved.eigenvalues).toExponential(3));
  
  // Test reconstruction accuracy
  console.log('\nReconstruction test:');
  const V = tf.tensor2d(improved.eigenvectors);
  const D = tf.diag(tf.tensor1d(improved.eigenvalues));
  const reconstructed = V.matMul(D).matMul(V.transpose());
  const reconError = await tf.norm(reconstructed.sub(laplacian)).data();
  console.log(`||V D V^T - A|| = ${reconError[0].toExponential(3)}`);
  
  // Test orthogonality
  const VtV = V.transpose().matMul(V);
  const I = tf.eye(laplacian.shape[0]);
  const orthoError = await tf.norm(VtV.sub(I)).data();
  console.log(`||V^T V - I|| = ${orthoError[0].toExponential(3)}`);
  
  // Compare on different test cases
  console.log('\n\nComparing on simple test matrix:');
  const testMatrix = tf.tensor2d([
    [2, -1, 0],
    [-1, 2, -1],
    [0, -1, 2]
  ]);
  
  console.log('\nOriginal Jacobi:');
  const test1 = jacobi_eigen_decomposition(testMatrix);
  console.log('Eigenvalues:', test1.eigenvalues.map(v => v.toFixed(6)));
  
  console.log('\nImproved Jacobi:');
  const test2 = improved_jacobi_eigen(testMatrix);
  console.log('Eigenvalues:', test2.eigenvalues.map(v => v.toFixed(6)));
  
  console.log('\nExpected (analytical):', [
    2 - 2 * Math.cos(Math.PI / 4),
    2 - 2 * Math.cos(2 * Math.PI / 4),
    2 - 2 * Math.cos(3 * Math.PI / 4)
  ].sort().map(v => v.toFixed(6)));
  
  // Cleanup
  X.dispose();
  affinity.dispose();
  laplacian.dispose();
  V.dispose();
  D.dispose();
  reconstructed.dispose();
  VtV.dispose();
  I.dispose();
  testMatrix.dispose();
}

testImprovedJacobi().catch(console.error);