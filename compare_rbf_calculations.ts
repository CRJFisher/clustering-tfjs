import * as tf from '@tensorflow/tfjs-node';
import { compute_rbf_affinity } from './src/utils/affinity';
import { SpectralClustering } from './src';

async function compareRBF() {
  console.log('Comparing RBF calculations with sklearn\n');
  
  // Load the failing test case
  const fixture = require('./test/fixtures/spectral/blobs_n2_rbf.json');
  
  console.log('Test case: blobs_n2_rbf');
  console.log('Shape:', fixture.X.length, 'x', fixture.X[0].length);
  console.log('Gamma:', fixture.params.gamma);
  console.log('Expected ARI: 0.95+');
  console.log('Current ARI: 0.064\n');
  
  // Convert to tensor
  const X = tf.tensor2d(fixture.X);
  
  // Compute our RBF affinity
  const ourAffinity = compute_rbf_affinity(X, fixture.params.gamma);
  const ourArray = await ourAffinity.array();
  
  // Sample some values to inspect
  console.log('Our RBF affinity matrix (sample values):');
  console.log('A[0,0]:', ourArray[0][0]);
  console.log('A[0,1]:', ourArray[0][1]);
  console.log('A[0,2]:', ourArray[0][2]);
  console.log('A[1,1]:', ourArray[1][1]);
  console.log('A[10,20]:', ourArray[10][20]);
  
  // Check matrix properties
  const minVal = await ourAffinity.min().data();
  const maxVal = await ourAffinity.max().data();
  const meanVal = await ourAffinity.mean().data();
  
  console.log('\nMatrix statistics:');
  console.log('Min:', minVal[0]);
  console.log('Max:', maxVal[0]);
  console.log('Mean:', meanVal[0]);
  
  // Check diagonal
  const diag = await tf.diag(ourAffinity).array();
  const diagSum = (diag as number[]).reduce((a, b) => a + b, 0);
  console.log('Diagonal sum:', diagSum, `(should be ${fixture.X.length})`);
  
  // Compute pairwise distances for a few sample points
  console.log('\nSample pairwise distances:');
  const x0 = fixture.X[0];
  const x1 = fixture.X[1];
  const x2 = fixture.X[2];
  
  const dist01 = Math.sqrt(x0.reduce((sum: number, val: number, i: number) => 
    sum + Math.pow(val - x1[i], 2), 0));
  const dist02 = Math.sqrt(x0.reduce((sum: number, val: number, i: number) => 
    sum + Math.pow(val - x2[i], 2), 0));
  
  console.log('||X[0] - X[1]||:', dist01);
  console.log('||X[0] - X[2]||:', dist02);
  
  // Manually compute RBF kernel values
  const gamma = fixture.params.gamma;
  const expectedA01 = Math.exp(-gamma * dist01 * dist01);
  const expectedA02 = Math.exp(-gamma * dist02 * dist02);
  
  console.log('\nExpected RBF values:');
  console.log('exp(-gamma * ||X[0]-X[1]||^2):', expectedA01);
  console.log('exp(-gamma * ||X[0]-X[2]||^2):', expectedA02);
  console.log('Actual A[0,1]:', ourArray[0][1]);
  console.log('Actual A[0,2]:', ourArray[0][2]);
  
  // Check gamma interpretation
  console.log('\n\nGamma analysis:');
  console.log('Fixture gamma:', fixture.params.gamma);
  console.log('Number of features:', fixture.X[0].length);
  console.log('Default gamma (1/n_features):', 1 / fixture.X[0].length);
  
  // Run clustering to see what happens
  console.log('\n\nRunning spectral clustering...');
  const model = new SpectralClustering({
    nClusters: fixture.params.nClusters,
    affinity: fixture.params.affinity,
    gamma: fixture.params.gamma,
    randomState: fixture.params.randomState
  });
  
  const labels = await model.fitPredict(fixture.X);
  
  // Count label distribution
  const labelCounts = new Map<number, number>();
  for (const label of labels) {
    labelCounts.set(label, (labelCounts.get(label) || 0) + 1);
  }
  console.log('Label distribution:', Object.fromEntries(labelCounts));
  console.log('Expected distribution: roughly 30/30 split');
  
  // Cleanup
  X.dispose();
  ourAffinity.dispose();
}

compareRBF().catch(console.error);