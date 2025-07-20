import * as tf from '@tensorflow/tfjs-node';
import { jacobi_eigen_decomposition } from './src/utils/laplacian';

// Test matrix with analytically known eigenvalues
// This is a tridiagonal matrix with eigenvalues: 2 - 2*cos(k*pi/(n+1)) for k=1..n
function createTridiagonalMatrix(n: number): tf.Tensor2D {
  const data: number[][] = Array(n).fill(0).map(() => Array(n).fill(0));
  
  for (let i = 0; i < n; i++) {
    data[i][i] = 2;
    if (i > 0) data[i][i-1] = -1;
    if (i < n-1) data[i][i+1] = -1;
  }
  
  return tf.tensor2d(data);
}

function analyticalEigenvalues(n: number): number[] {
  const eigenvalues: number[] = [];
  for (let k = 1; k <= n; k++) {
    eigenvalues.push(2 - 2 * Math.cos(k * Math.PI / (n + 1)));
  }
  return eigenvalues.sort((a, b) => a - b);
}

async function compareEigenvalues() {
  console.log('Comparing Jacobi solver with analytical eigenvalues\n');
  
  for (const n of [5, 10, 15]) {
    console.log(`\nMatrix size: ${n}x${n}`);
    console.log('=' .repeat(50));
    
    const matrix = createTridiagonalMatrix(n);
    const expected = analyticalEigenvalues(n);
    const result = jacobi_eigen_decomposition(matrix);
    
    console.log('Index | Expected    | Computed    | Error');
    console.log('-'.repeat(50));
    
    let maxError = 0;
    for (let i = 0; i < n; i++) {
      const error = Math.abs(expected[i] - result.eigenvalues[i]);
      maxError = Math.max(maxError, error);
      console.log(
        `${i.toString().padStart(5)} | ${expected[i].toFixed(8)} | ${result.eigenvalues[i].toFixed(8)} | ${error.toExponential(2)}`
      );
    }
    
    console.log(`\nMax error: ${maxError.toExponential(3)}`);
    
    // Test orthogonality of eigenvectors
    const V = tf.tensor2d(result.eigenvectors);
    const VtV = V.transpose().matMul(V);
    const I = tf.eye(n);
    const orthoError = tf.norm(VtV.sub(I)).dataSync()[0];
    console.log(`Orthogonality error (||V^T V - I||): ${orthoError.toExponential(3)}`);
    
    matrix.dispose();
    V.dispose();
    VtV.dispose();
    I.dispose();
  }
  
  // Test on a normalized Laplacian from the fixtures
  console.log('\n\nTesting on actual fixture data:');
  console.log('=' .repeat(50));
  
  const fixture = require('./test/fixtures/spectral/blobs_n3_knn.json');
  const { compute_knn_affinity } = require('./src/utils/affinity');
  const { normalised_laplacian } = require('./src/utils/laplacian');
  
  const X = tf.tensor2d(fixture.X);
  const affinity = compute_knn_affinity(X, fixture.params.nNeighbors, true);
  const laplacian = normalised_laplacian(affinity);
  
  console.log('Running Jacobi with different settings:');
  
  for (const config of [
    { maxIterations: 100, tolerance: 1e-10 },
    { maxIterations: 1000, tolerance: 1e-10 },
    { maxIterations: 2000, tolerance: 1e-12 },
    { maxIterations: 5000, tolerance: 1e-14 },
  ]) {
    const start = Date.now();
    const result = jacobi_eigen_decomposition(laplacian, config);
    const time = Date.now() - start;
    
    const smallestEigs = result.eigenvalues.slice(0, 5);
    console.log(`\nConfig: ${JSON.stringify(config)}`);
    console.log(`Time: ${time}ms`);
    console.log(`First 5 eigenvalues: ${smallestEigs.map(v => v.toExponential(3)).join(', ')}`);
  }
  
  X.dispose();
  affinity.dispose();
  laplacian.dispose();
}

compareEigenvalues().catch(console.error);