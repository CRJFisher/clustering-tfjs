import * as tf from '@tensorflow/tfjs';
import { compute_knn_affinity, compute_rbf_affinity } from './dist/utils/affinity';
import { normalised_laplacian } from './dist/utils/laplacian';
import * as fs from 'fs';
import * as path from 'path';

async function debugMoonsRegression() {
  console.log('=== Debugging Moons Regression ===\n');
  
  // Test both moons datasets that regressed
  const datasets = ['moons_n2_knn.json', 'moons_n2_rbf.json'];
  
  for (const dataset of datasets) {
    console.log(`\n--- Testing ${dataset} ---`);
    const fixture = JSON.parse(
      fs.readFileSync(path.join(__dirname, 'test/fixtures/spectral', dataset), 'utf-8')
    );
    
    const X = tf.tensor2d(fixture.X);
    console.log('Data shape:', X.shape);
    
    // Compute affinity based on type
    let affinity;
    if (dataset.includes('knn')) {
      affinity = compute_knn_affinity(X, fixture.params.nNeighbors, true);
    } else {
      affinity = compute_rbf_affinity(X, fixture.params.gamma);
    }
    
    // Get degrees to understand connectivity
    const affinityArray = await affinity.array();
    const degrees = affinityArray.map(row => row.reduce((sum, val) => sum + val, 0));
    
    console.log('\nConnectivity analysis:');
    console.log('Min degree:', Math.min(...degrees).toFixed(2));
    console.log('Max degree:', Math.max(...degrees).toFixed(2));
    console.log('Avg degree:', (degrees.reduce((a, b) => a + b) / degrees.length).toFixed(2));
    
    // Check if graph has disconnected components
    const zeroDegrees = degrees.filter(d => d < 0.1).length;
    console.log('Near-zero degree nodes:', zeroDegrees);
    
    // Get Laplacian eigenvalues to check for disconnected components
    const { laplacian, sqrtDegrees } = normalised_laplacian(affinity, true);
    
    // Check sqrtDegrees for issues
    const sqrtDegreesArray = await sqrtDegrees.array() as number[];
    const minSqrtDeg = Math.min(...sqrtDegreesArray);
    const maxSqrtDeg = Math.max(...sqrtDegreesArray);
    console.log(`\nsqrtDegrees (D^(-1/2)) range: [${minSqrtDeg.toFixed(6)}, ${maxSqrtDeg.toFixed(6)}]`);
    
    // For well-connected graphs, dividing by very small sqrtDegrees might amplify noise
    const verySmallDegrees = sqrtDegreesArray.filter(d => d > 10).length;
    console.log('Very small degrees (sqrtDeg > 10):', verySmallDegrees);
    
    // Cleanup
    X.dispose();
    affinity.dispose();
    laplacian.dispose();
    sqrtDegrees.dispose();
  }
  
  console.log('\n=== Analysis ===');
  console.log('For well-connected graphs (like moons), eigenvector recovery might:');
  console.log('1. Amplify numerical noise by dividing by small sqrt(degrees)');
  console.log('2. Not be appropriate since there are no disconnected components');
  console.log('3. sklearn might detect this and skip recovery for well-connected graphs');
}

debugMoonsRegression().catch(console.error);