import * as tf from '@tensorflow/tfjs-node';
import { compute_rbf_affinity } from './src/utils/affinity';
import { SpectralClustering } from './src';

async function investigateGamma() {
  console.log('Investigating gamma parameter issue\n');
  
  const fixture = require('./test/fixtures/spectral/blobs_n2_rbf.json');
  
  console.log('Dataset info:');
  console.log('- Shape:', fixture.X.length, 'x', fixture.X[0].length);
  console.log('- Fixture gamma:', fixture.params.gamma);
  console.log('- Default gamma (1/n_features):', 1 / fixture.X[0].length, '(0.5)');
  
  // Analyze distances in the dataset
  const X = tf.tensor2d(fixture.X);
  const distances: number[] = [];
  
  // Sample some pairwise distances
  for (let i = 0; i < Math.min(10, fixture.X.length); i++) {
    for (let j = i + 1; j < Math.min(10, fixture.X.length); j++) {
      const xi = fixture.X[i];
      const xj = fixture.X[j];
      const dist_sq = xi.reduce((sum: number, val: number, k: number) => 
        sum + Math.pow(val - xj[k], 2), 0);
      distances.push(Math.sqrt(dist_sq));
    }
  }
  
  console.log('\nSample distances in dataset:');
  console.log('Min:', Math.min(...distances).toFixed(3));
  console.log('Max:', Math.max(...distances).toFixed(3));
  console.log('Mean:', (distances.reduce((a, b) => a + b) / distances.length).toFixed(3));
  
  // Test with different gamma values
  console.log('\n\nTesting different gamma values:');
  
  for (const testGamma of [0.01, 0.1, 0.5, 1.0, 2.0]) {
    console.log(`\nGamma = ${testGamma}:`);
    
    const affinity = compute_rbf_affinity(X, testGamma);
    const A = await affinity.array();
    
    // Count non-trivial affinity values
    let count = 0;
    let sum = 0;
    for (let i = 0; i < A.length; i++) {
      for (let j = 0; j < A[i].length; j++) {
        if (i !== j) {
          if (A[i][j] > 0.01) count++;
          sum += A[i][j];
        }
      }
    }
    
    console.log(`- Non-trivial (>0.01) connections: ${count}/${A.length * (A.length - 1)}`);
    console.log(`- Mean off-diagonal affinity: ${(sum / (A.length * (A.length - 1))).toFixed(6)}`);
    
    affinity.dispose();
  }
  
  // Check if gamma is being passed correctly
  console.log('\n\nChecking SpectralClustering gamma handling:');
  const model = new SpectralClustering({
    nClusters: 2,
    affinity: 'rbf',
    gamma: fixture.params.gamma,
    randomState: 42
  });
  
  console.log('Model params:', model.params);
  
  X.dispose();
}

investigateGamma().catch(console.error);