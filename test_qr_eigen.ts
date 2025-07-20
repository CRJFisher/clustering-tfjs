import * as tf from '@tensorflow/tfjs-node';
import { qr_eigen_decomposition } from './src/utils/eigen_qr';
import { jacobi_eigen_decomposition } from './src/utils/laplacian';

async function testQREigen() {
  console.log('Testing QR-based eigensolver\n');
  
  // Test 1: Simple symmetric matrix
  console.log('Test 1: Simple 3x3 symmetric matrix');
  const A1 = tf.tensor2d([
    [4, -1, -1],
    [-1, 3, -1],
    [-1, -1, 2]
  ]);
  
  console.log('\nJacobi solver:');
  const jacobi1 = jacobi_eigen_decomposition(A1);
  console.log('Eigenvalues:', jacobi1.eigenvalues);
  
  console.log('\nQR solver:');
  const qr1 = qr_eigen_decomposition(A1);
  console.log('Eigenvalues:', qr1.eigenvalues);
  
  // Test 2: Normalized Laplacian from fixture
  console.log('\n\nTest 2: Normalized Laplacian from fixture');
  const fixture = require('./test/fixtures/spectral/blobs_n3_knn.json');
  const { compute_knn_affinity } = require('./src/utils/affinity');
  const { normalised_laplacian } = require('./src/utils/laplacian');
  
  const X = tf.tensor2d(fixture.X);
  const affinity = compute_knn_affinity(X, fixture.params.nNeighbors, true);
  const laplacian = normalised_laplacian(affinity);
  
  console.log('Matrix size:', laplacian.shape);
  
  console.log('\nJacobi solver (2000 iterations):');
  const start1 = Date.now();
  const jacobi2 = jacobi_eigen_decomposition(laplacian, { maxIterations: 2000 });
  const time1 = Date.now() - start1;
  console.log(`Time: ${time1}ms`);
  console.log('First 5 eigenvalues:', jacobi2.eigenvalues.slice(0, 5).map(v => v.toExponential(3)));
  
  console.log('\nQR solver:');
  const start2 = Date.now();
  const qr2 = qr_eigen_decomposition(laplacian);
  const time2 = Date.now() - start2;
  console.log(`Time: ${time2}ms`);
  console.log('First 5 eigenvalues:', qr2.eigenvalues.slice(0, 5).map(v => v.toExponential(3)));
  
  // Test orthogonality
  console.log('\nOrthogonality test for QR eigenvectors:');
  const V = tf.tensor2d(qr2.eigenvectors);
  const VtV = V.transpose().matMul(V);
  const I = tf.eye(laplacian.shape[0]);
  const orthoError = await tf.norm(VtV.sub(I)).data();
  console.log(`||V^T V - I|| = ${orthoError[0].toExponential(3)}`);
  
  // Test reconstruction
  console.log('\nReconstruction test:');
  const D = tf.diag(tf.tensor1d(qr2.eigenvalues));
  const reconstructed = V.matMul(D).matMul(V.transpose());
  const reconError = await tf.norm(reconstructed.sub(laplacian)).data();
  console.log(`||V D V^T - A|| = ${reconError[0].toExponential(3)}`);
  
  // Cleanup
  A1.dispose();
  X.dispose();
  affinity.dispose();
  laplacian.dispose();
  V.dispose();
  VtV.dispose();
  I.dispose();
  D.dispose();
  reconstructed.dispose();
}

testQREigen().catch(console.error);