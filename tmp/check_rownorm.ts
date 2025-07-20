import fs from 'fs';
import * as tf from '@tensorflow/tfjs-node';
import { SpectralClustering } from '../src';
import { normalised_laplacian, smallest_eigenvectors } from '../src/utils/laplacian';

const fix = JSON.parse(fs.readFileSync(process.argv[2],'utf-8'));
const X = tf.tensor2d(fix.X as number[][], undefined, 'float32');
const A = SpectralClustering.computeAffinityMatrix(X, {
  affinity: fix.params.affinity,
  nNeighbors: fix.params.nNeighbors,
  gamma: fix.params.gamma,
  nClusters: fix.params.nClusters,
} as any);
const L = normalised_laplacian(A);
const U_full = smallest_eigenvectors(L, fix.params.nClusters);
const trivial = U_full.shape[1] - fix.params.nClusters;
const U = tf.slice(U_full,[0,trivial],[-1, fix.params.nClusters]);
const norms = U.norm('euclidean',1);
(async()=>{
  const arr = await norms.array() as number[];
  console.log('minNorm', Math.min(...arr), 'maxNorm', Math.max(...arr));
})();

