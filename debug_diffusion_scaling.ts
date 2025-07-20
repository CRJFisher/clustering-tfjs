import * as tf from '@tensorflow/tfjs-node';
import { SpectralClustering } from './src';
import fs from 'fs';

// Test if diffusion map scaling is working
async function debugDiffusionScaling() {
  // Use circles_n2_knn as test case
  const fixture = require('./test/fixtures/spectral/circles_n2_knn.json');
  
  console.log('Testing diffusion map scaling implementation');
  console.log('Dataset: circles_n2_knn\n');
  
  // Create a simple test case first
  const simpleX = [
    [0, 0],
    [1, 0],
    [0, 1],
    [1, 1],
  ];
  
  const model = new SpectralClustering({
    nClusters: 2,
    affinity: 'nearest_neighbors',
    nNeighbors: 2,
    randomState: 42
  });
  
  console.log('Running on simple 4-point test case...');
  await model.fit(simpleX);
  console.log('Labels:', model.labels_);
  
  // Now test on actual fixture
  console.log('\nRunning on circles_n2_knn fixture...');
  const model2 = new SpectralClustering({
    nClusters: fixture.params.nClusters,
    affinity: fixture.params.affinity,
    nNeighbors: fixture.params.nNeighbors,
    randomState: fixture.params.randomState
  });
  
  await model2.fit(fixture.X);
  console.log('Labels (first 10):', model2.labels_?.slice(0, 10));
  
  model.dispose();
  model2.dispose();
}

debugDiffusionScaling().catch(console.error);