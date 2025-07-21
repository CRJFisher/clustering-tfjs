import * as tf from '@tensorflow/tfjs-node';
import { compute_knn_affinity } from './dist/utils/affinity';
import { normalised_laplacian } from './dist/utils/laplacian';
import * as fs from 'fs';
import * as path from 'path';

async function compareOurAffinity() {
  console.log('=== Comparing our affinity matrix with sklearn ===\n');
  
  // Load moons dataset
  const fixture = JSON.parse(
    fs.readFileSync(path.join(__dirname, 'test/fixtures/spectral/moons_n2_knn.json'), 'utf-8')
  );
  
  const X = tf.tensor2d(fixture.X);
  const n = X.shape[0];
  
  // Compute our affinity matrix
  const affinity = compute_knn_affinity(X, fixture.params.nNeighbors, true);
  const affinityArray = await affinity.array();
  
  console.log('Our affinity matrix:');
  console.log(`  Shape: [${affinity.shape}]`);
  
  // Check diagonal
  let diagSum = 0;
  for (let i = 0; i < n; i++) {
    diagSum += affinityArray[i][i];
  }
  console.log(`  Diagonal sum: ${diagSum.toFixed(6)}`);
  
  // Check symmetry
  let isSymmetric = true;
  for (let i = 0; i < n && isSymmetric; i++) {
    for (let j = i + 1; j < n; j++) {
      if (Math.abs(affinityArray[i][j] - affinityArray[j][i]) > 1e-10) {
        isSymmetric = false;
        break;
      }
    }
  }
  console.log(`  Is symmetric: ${isSymmetric}`);
  
  // Check degrees
  const degrees = affinityArray.map(row => row.reduce((sum, val) => sum + val, 0));
  console.log(`  Degree range: [${Math.min(...degrees).toFixed(2)}, ${Math.max(...degrees).toFixed(2)}]`);
  console.log(`  Mean degree: ${(degrees.reduce((a, b) => a + b) / n).toFixed(2)}`);
  
  // Check first row
  console.log('\nFirst row non-zero entries:');
  const row0 = affinityArray[0];
  for (let j = 0; j < Math.min(15, n); j++) {
    if (row0[j] > 0) {
      console.log(`    A[0, ${j}] = ${row0[j].toFixed(6)}`);
    }
  }
  
  // Check unique values
  const uniqueVals = new Set<string>();
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      if (affinityArray[i][j] > 0) {
        uniqueVals.add(affinityArray[i][j].toFixed(6));
      }
    }
  }
  console.log(`\n  Unique non-zero values: ${uniqueVals.size}`);
  if (uniqueVals.size < 10) {
    console.log(`    Values: ${Array.from(uniqueVals).sort().join(', ')}`);
  }
  
  // Test normalized Laplacian
  const laplacian = normalised_laplacian(affinity);
  const laplacianArray = await laplacian.array();
  
  console.log('\nNormalized Laplacian check:');
  
  // L * 1
  const ones = new Array(n).fill(1);
  const L_times_ones = new Array(n).fill(0);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      L_times_ones[i] += laplacianArray[i][j] * ones[j];
    }
  }
  const norm1 = Math.sqrt(L_times_ones.reduce((sum, val) => sum + val * val, 0));
  console.log(`  ||L * 1|| = ${norm1.toFixed(10)}`);
  
  // L * (1/sqrt(n))
  const constVec = new Array(n).fill(1 / Math.sqrt(n));
  const L_times_const = new Array(n).fill(0);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      L_times_const[i] += laplacianArray[i][j] * constVec[j];
    }
  }
  const normConst = Math.sqrt(L_times_const.reduce((sum, val) => sum + val * val, 0));
  console.log(`  ||L * (1/sqrt(n))|| = ${normConst.toFixed(10)}`);
  
  // Clean up
  X.dispose();
  affinity.dispose();
  laplacian.dispose();
}

compareOurAffinity().catch(console.error);