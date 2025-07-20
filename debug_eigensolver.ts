import * as tf from '@tensorflow/tfjs-node';
import { improved_jacobi_eigen } from './src/utils/eigen_improved';

async function debugEigensolver() {
  // Load a failing fixture
  const fixture = require('./test/fixtures/spectral/circles_n2_rbf.json');
  const { compute_rbf_affinity } = require('./src/utils/affinity');
  const { normalised_laplacian } = require('./src/utils/laplacian');
  
  const X = tf.tensor2d(fixture.X);
  const affinity = compute_rbf_affinity(X, fixture.params.gamma);
  const laplacian = normalised_laplacian(affinity);
  
  console.log('Debugging circles_n2_rbf (ARI = 0.8689)');
  console.log('Matrix size:', laplacian.shape);
  console.log('Expected labels:', fixture.labels.slice(0, 10), '...');
  
  // Run eigensolver
  console.log('\nRunning improved Jacobi eigensolver...');
  const result = improved_jacobi_eigen(laplacian, {
    isPSD: true,
    maxIterations: 5000,
    tolerance: 1e-14,
  });
  
  console.log('\nEigenvalue distribution:');
  console.log('First 10:', result.eigenvalues.slice(0, 10).map(v => v.toExponential(3)));
  console.log('Last 10:', result.eigenvalues.slice(-10).map(v => v.toExponential(3)));
  
  // Count near-zero eigenvalues
  const counts = {
    zero: 0,
    tiny: 0,
    small: 0,
    medium: 0,
    large: 0
  };
  
  for (const val of result.eigenvalues) {
    if (val === 0) counts.zero++;
    else if (val < 1e-5) counts.tiny++;
    else if (val < 0.1) counts.small++;
    else if (val < 1.0) counts.medium++;
    else counts.large++;
  }
  
  console.log('\nEigenvalue counts:');
  console.log(`Zero (= 0): ${counts.zero}`);
  console.log(`Tiny (< 1e-5): ${counts.tiny}`);
  console.log(`Small (< 0.1): ${counts.small}`);
  console.log(`Medium (< 1.0): ${counts.medium}`);
  console.log(`Large (>= 1.0): ${counts.large}`);
  
  // Check spectral gap
  const nonZeroEigs = result.eigenvalues.filter(v => v > 1e-5);
  if (nonZeroEigs.length >= fixture.params.nClusters) {
    const gap = nonZeroEigs[fixture.params.nClusters - 1] - nonZeroEigs[fixture.params.nClusters - 2];
    console.log(`\nSpectral gap at k=${fixture.params.nClusters}: ${gap.toExponential(3)}`);
  }
  
  // Run clustering to see what's happening
  const { SpectralClustering } = require('./src');
  const params: any = {
    nClusters: fixture.params.nClusters,
    affinity: fixture.params.affinity,
    randomState: fixture.params.randomState
  };
  if (fixture.params.gamma != null) params.gamma = fixture.params.gamma;
  
  const model = new SpectralClustering(params);
  const labels = await model.fitPredict(fixture.X);
  
  console.log('\nClustering results:');
  console.log('Our labels (first 10):', labels.slice(0, 10));
  console.log('Expected (first 10):', fixture.labels.slice(0, 10));
  
  // Count label distribution
  const labelCounts = new Map<number, number>();
  for (const label of labels) {
    labelCounts.set(label, (labelCounts.get(label) || 0) + 1);
  }
  console.log('\nLabel distribution:', Object.fromEntries(labelCounts));
  
  X.dispose();
  affinity.dispose();
  laplacian.dispose();
}

debugEigensolver().catch(console.error);