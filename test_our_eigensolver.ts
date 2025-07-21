import * as tf from '@tensorflow/tfjs-node';
import { compute_knn_affinity } from './dist/utils/affinity';
import { normalised_laplacian } from './dist/utils/laplacian';
import * as fs from 'fs';
import * as path from 'path';

async function testOurEigensolver() {
  console.log('=== Testing our eigensolver vs numpy ===\n');
  
  // Test with simple example first
  console.log('Test 1: Simple 3x3 matrix');
  const simple = tf.tensor2d([
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1]
  ]);
  
  // This should have eigenvalues [1, 1, 1] and identity eigenvectors
  const simpleEig = tf.linalg.eigh(simple);
  console.log('Eigenvalues:', await simpleEig.values.array());
  console.log('Eigenvectors:');
  console.log(await simpleEig.vectors.array());
  
  simpleEig.values.dispose();
  simpleEig.vectors.dispose();
  simple.dispose();
  
  // Now test with moons Laplacian
  console.log('\nTest 2: Moons Laplacian');
  const fixture = JSON.parse(
    fs.readFileSync(path.join(__dirname, 'test/fixtures/spectral/moons_n2_knn.json'), 'utf-8')
  );
  
  const X = tf.tensor2d(fixture.X);
  const affinity = compute_knn_affinity(X, fixture.params.nNeighbors, true);
  const laplacian = normalised_laplacian(affinity);
  
  // Use TensorFlow's built-in eigensolver
  const eig = tf.linalg.eigh(laplacian);
  const eigenvalues = await eig.values.array();
  const eigenvectors = await eig.vectors.array();
  
  console.log('\nFirst 5 eigenvalues:', eigenvalues.slice(0, 5));
  
  // Check first eigenvector
  const firstEigenvector = eigenvectors.map(row => row[0]);
  const firstStd = Math.sqrt(firstEigenvector.reduce((sum, v) => {
    const mean = firstEigenvector.reduce((a, b) => a + b) / firstEigenvector.length;
    return sum + Math.pow(v - mean, 2);
  }, 0) / firstEigenvector.length);
  
  console.log('\nFirst eigenvector analysis:');
  console.log('  Unique values:', new Set(firstEigenvector.map(v => v.toFixed(6))).size);
  console.log('  Std deviation:', firstStd);
  console.log('  Is constant:', firstStd < 1e-10);
  console.log('  First 5 values:', firstEigenvector.slice(0, 5));
  
  // Clean up
  eig.values.dispose();
  eig.vectors.dispose();
  X.dispose();
  affinity.dispose();
  laplacian.dispose();
}

testOurEigensolver().catch(console.error);