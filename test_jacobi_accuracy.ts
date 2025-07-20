import * as tf from '@tensorflow/tfjs-node';
import { jacobi_eigen_decomposition } from './src/utils/laplacian';

async function testJacobiAccuracy() {
  console.log('Testing Jacobi eigensolver accuracy\n');
  
  // Test 1: Simple symmetric matrix with known eigenvalues
  console.log('Test 1: Simple 3x3 symmetric matrix');
  const A1 = tf.tensor2d([
    [4, -1, -1],
    [-1, 3, -1],
    [-1, -1, 2]
  ]);
  
  const result1 = jacobi_eigen_decomposition(A1);
  console.log('Eigenvalues:', result1.eigenvalues);
  console.log('Expected: [1, 3, 5] (approximately)');
  
  // Test 2: Identity matrix (should have all eigenvalues = 1)
  console.log('\nTest 2: 5x5 Identity matrix');
  const A2 = tf.eye(5);
  const result2 = jacobi_eigen_decomposition(A2);
  console.log('Eigenvalues:', result2.eigenvalues);
  console.log('Expected: [1, 1, 1, 1, 1]');
  
  // Test 3: Matrix with known small eigenvalues
  console.log('\nTest 3: Matrix with small eigenvalues');
  const A3 = tf.tensor2d([
    [1e-10, 0, 0],
    [0, 1e-8, 0],
    [0, 0, 1e-6]
  ]);
  const result3 = jacobi_eigen_decomposition(A3);
  console.log('Eigenvalues:', result3.eigenvalues);
  console.log('Expected: [1e-10, 1e-8, 1e-6]');
  
  // Test 4: Test convergence on a harder matrix
  console.log('\nTest 4: Convergence test on normalized Laplacian-like matrix');
  const A4 = tf.tensor2d([
    [1.0, -0.5, -0.3, 0.0],
    [-0.5, 1.0, -0.4, -0.1],
    [-0.3, -0.4, 1.0, -0.3],
    [0.0, -0.1, -0.3, 0.4]
  ]);
  
  console.log('\nTesting different iteration counts:');
  for (const maxIter of [100, 500, 1000, 2000, 5000]) {
    const start = Date.now();
    const result = jacobi_eigen_decomposition(A4, { maxIterations: maxIter });
    const time = Date.now() - start;
    
    // Check off-diagonal norm
    const matrix = await A4.array();
    const n = matrix.length;
    const V = result.eigenvectors;
    
    // Reconstruct A = V * D * V^T and check error
    const D = tf.diag(tf.tensor1d(result.eigenvalues));
    const Vt = tf.tensor2d(V);
    const reconstructed = Vt.matMul(D).matMul(Vt.transpose());
    const error = tf.norm(reconstructed.sub(A4)).dataSync()[0];
    
    console.log(`Iterations: ${maxIter}, Time: ${time}ms, Reconstruction error: ${error.toExponential(3)}`);
    console.log(`  Smallest eigenvalue: ${Math.min(...result.eigenvalues).toExponential(3)}`);
    
    reconstructed.dispose();
    D.dispose();
    Vt.dispose();
  }
  
  // Test 5: Compare with a matrix from actual spectral clustering
  console.log('\nTest 5: Actual normalized Laplacian from k-NN graph');
  const points = tf.randomNormal([20, 2]);
  const { compute_knn_affinity } = require('./src/utils/affinity');
  const { normalised_laplacian } = require('./src/utils/laplacian');
  
  const affinity = compute_knn_affinity(points, 5, true);
  const laplacian = normalised_laplacian(affinity);
  
  console.log('Matrix size:', laplacian.shape);
  const result5 = jacobi_eigen_decomposition(laplacian);
  const sortedEigs = [...result5.eigenvalues].sort((a, b) => a - b);
  console.log('First 5 eigenvalues:', sortedEigs.slice(0, 5).map(v => v.toExponential(3)));
  console.log('Last 5 eigenvalues:', sortedEigs.slice(-5).map(v => v.toExponential(3)));
  
  // Cleanup
  A1.dispose();
  A2.dispose();
  A3.dispose();
  A4.dispose();
  points.dispose();
  affinity.dispose();
  laplacian.dispose();
}

testJacobiAccuracy().catch(console.error);