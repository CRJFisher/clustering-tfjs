import * as tf from '@tensorflow/tfjs-node';
import { SpectralClustering } from './dist/clustering/spectral';
import { compute_knn_affinity, compute_rbf_affinity } from './dist/utils/affinity';
import * as fs from 'fs';
import * as path from 'path';

// Quick test to see if we actually had 6/12 passing before
async function testWithoutDiffusion() {
  console.log('=== Testing specific fixtures that user mentioned were passing ===\n');
  
  // Test the fixtures that were supposedly passing
  const fixtures = [
    'moons_n2_knn.json',
    'moons_n2_rbf.json', 
    'blobs_n2_knn.json',
    'circles_n2_knn.json'
  ];
  
  for (const fixtureName of fixtures) {
    const fixture = JSON.parse(
      fs.readFileSync(path.join(__dirname, 'test/fixtures/spectral', fixtureName), 'utf-8')
    );
    
    const X_tf = tf.tensor2d(fixture.X);
    const y_true = fixture.labels;
    
    const sc = new SpectralClustering({
      nClusters: fixture.params.nClusters,
      randomState: fixture.params.randomState
    });
    
    let affinity: tf.Tensor2D;
    if (fixtureName.includes('knn')) {
      affinity = compute_knn_affinity(X_tf, fixture.params.nNeighbors, true);
    } else {
      affinity = compute_rbf_affinity(X_tf, fixture.params.gamma);
    }
    
    const predictions = await sc.fitPredict(affinity);
    const y_pred = Array.isArray(predictions) ? predictions : await predictions.array();
    
    console.log(`${fixtureName}: First 10 predictions: ${y_pred.slice(0, 10)}`);
    console.log(`           Expected:         ${y_true.slice(0, 10)}`);
    
    X_tf.dispose();
    affinity.dispose();
  }
}

testWithoutDiffusion().catch(console.error);