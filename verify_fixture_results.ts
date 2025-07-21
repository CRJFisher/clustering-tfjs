import * as tf from '@tensorflow/tfjs-node';
import { SpectralClustering } from './dist/clustering/spectral';
import { compute_knn_affinity, compute_rbf_affinity } from './dist/utils/affinity';
// Simple ARI implementation for verification
function adjustedRandIndex(labels_true: number[], labels_pred: number[]): number {
  const n = labels_true.length;
  if (n !== labels_pred.length) {
    throw new Error('Label arrays must have same length');
  }
  
  // Create contingency table
  const labelMap = new Map<string, number>();
  for (let i = 0; i < n; i++) {
    const key = `${labels_true[i]},${labels_pred[i]}`;
    labelMap.set(key, (labelMap.get(key) || 0) + 1);
  }
  
  // Calculate sums
  const a_sum = new Map<number, number>();
  const b_sum = new Map<number, number>();
  
  for (let i = 0; i < n; i++) {
    a_sum.set(labels_true[i], (a_sum.get(labels_true[i]) || 0) + 1);
    b_sum.set(labels_pred[i], (b_sum.get(labels_pred[i]) || 0) + 1);
  }
  
  // Calculate index
  let sum_comb = 0;
  labelMap.forEach((count) => {
    if (count >= 2) {
      sum_comb += (count * (count - 1)) / 2;
    }
  });
  
  let sum_a = 0;
  a_sum.forEach((count) => {
    if (count >= 2) {
      sum_a += (count * (count - 1)) / 2;
    }
  });
  
  let sum_b = 0;
  b_sum.forEach((count) => {
    if (count >= 2) {
      sum_b += (count * (count - 1)) / 2;
    }
  });
  
  const expected_index = (sum_a * sum_b) / ((n * (n - 1)) / 2);
  const max_index = (sum_a + sum_b) / 2;
  
  if (max_index === expected_index) {
    return 0;
  }
  
  return (sum_comb - expected_index) / (max_index - expected_index);
}
import * as fs from 'fs';
import * as path from 'path';

interface FixtureData {
  X: number[][];
  labels: number[];
  params: {
    nClusters: number;
    nNeighbors?: number;
    gamma?: number;
    randomState: number;
  };
}

async function verifyFixtureResults() {
  console.log('=== Verifying fixture results with diffusion scaling ===\n');
  
  // Test key fixtures that were problematic
  const fixtures = [
    'blobs_n2_knn.json',
    'blobs_n2_rbf.json', 
    'moons_n2_knn.json',
    'moons_n2_rbf.json',
    'circles_n2_knn.json',
    'circles_n2_rbf.json'
  ];
  
  let passCount = 0;
  let failCount = 0;
  
  for (const fixtureName of fixtures) {
    const fixture: FixtureData = JSON.parse(
      fs.readFileSync(path.join(__dirname, 'test/fixtures/spectral', fixtureName), 'utf-8')
    );
    
    const X_tf = tf.tensor2d(fixture.X);
    const y_true = fixture.labels;
    
    // Create SpectralClustering instance
    const sc = new SpectralClustering({
      nClusters: fixture.params.nClusters,
      randomState: fixture.params.randomState
    });
    
    // Compute affinity matrix based on type
    let affinity: tf.Tensor2D;
    if (fixtureName.includes('knn')) {
      affinity = compute_knn_affinity(X_tf, fixture.params.nNeighbors!, true);
    } else {
      affinity = compute_rbf_affinity(X_tf, fixture.params.gamma!);
    }
    
    // Fit and predict
    const predictions = await sc.fitPredict(affinity);
    const y_pred = Array.isArray(predictions) ? predictions : await predictions.array();
    
    // Calculate ARI
    const ari = adjustedRandIndex(y_true, y_pred);
    
    // Expected ARI = 1.0 for all fixtures with sklearn
    const passed = ari >= 0.95; // Allow small tolerance
    
    console.log(`${fixtureName}: ARI = ${ari.toFixed(4)} ${passed ? '✓ PASS' : '✗ FAIL'}`);
    
    if (passed) passCount++;
    else failCount++;
    
    // Clean up
    X_tf.dispose();
    affinity.dispose();
    if (!Array.isArray(predictions) && predictions.dispose) {
      predictions.dispose();
    }
  }
  
  console.log(`\nSummary: ${passCount} passed, ${failCount} failed`);
  
  // Check specific cases that were problematic before
  console.log('\n=== Key cases ===');
  console.log('blobs_n2_knn: Should now pass (was 0.088 with recovery)');
  console.log('moons_n2_knn: Should pass (was failing with recovery)');
  console.log('moons_n2_rbf: Should pass (was failing with recovery)');
}

verifyFixtureResults().catch(console.error);