import * as tf from '@tensorflow/tfjs-node';
import { compute_rbf_affinity } from './src/utils/affinity';

async function testRBFSimple() {
  console.log('Testing RBF implementation\n');
  
  // Simple 2D points
  const points = [
    [0, 0],
    [1, 0],
    [0, 1],
    [10, 10]
  ];
  
  const X = tf.tensor2d(points);
  const gamma = 1.0;
  
  console.log('Points:');
  points.forEach((p, i) => console.log(`  X[${i}] = [${p}]`));
  console.log('\nGamma:', gamma);
  
  // Compute RBF affinity
  const affinity = compute_rbf_affinity(X, gamma);
  const A = await affinity.array();
  
  console.log('\nRBF Affinity Matrix:');
  A.forEach((row, i) => {
    console.log(`Row ${i}:`, row.map(v => v.toFixed(6)).join(', '));
  });
  
  // Manual calculation for verification
  console.log('\nManual calculations:');
  console.log('d(X[0], X[1])^2 = (0-1)^2 + (0-0)^2 = 1');
  console.log('exp(-gamma * 1) = exp(-1) =', Math.exp(-1));
  console.log('A[0,1] from matrix:', A[0][1]);
  
  console.log('\nd(X[0], X[3])^2 = (0-10)^2 + (0-10)^2 = 200');
  console.log('exp(-gamma * 200) = exp(-200) =', Math.exp(-200));
  console.log('A[0,3] from matrix:', A[0][3]);
  
  // Check the failing test
  console.log('\n\nChecking blobs_n2_rbf data:');
  const fixture = require('./test/fixtures/spectral/blobs_n2_rbf.json');
  
  // Look at a few sample points
  console.log('First few points:');
  for (let i = 0; i < 3; i++) {
    console.log(`  X[${i}] = [${fixture.X[i]}]`);
  }
  
  // Compute distances between first few points
  const x0 = fixture.X[0];
  const x1 = fixture.X[1];
  const dist_sq = x0.reduce((sum: number, val: number, i: number) => 
    sum + Math.pow(val - x1[i], 2), 0);
  
  console.log('\nDistance calculations:');
  console.log('||X[0] - X[1]||^2 =', dist_sq);
  console.log('With gamma=1: exp(-1 * ' + dist_sq + ') =', Math.exp(-dist_sq));
  
  // The issue might be the gamma value
  console.log('\nSuspicion: gamma might be too large for this dataset');
  console.log('Typical distances squared in this dataset are ~10-20');
  console.log('With gamma=1, exp(-15) ≈', Math.exp(-15), '(very small!)');
  
  X.dispose();
  affinity.dispose();
}

testRBFSimple().catch(console.error);