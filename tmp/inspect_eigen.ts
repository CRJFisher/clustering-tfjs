import fs from 'fs';
import * as tf from '@tensorflow/tfjs-node';
import { SpectralClustering } from '../src';
import { normalised_laplacian, jacobi_eigen_decomposition } from '../src/utils/laplacian';

const fixturePath = process.argv[2];
const fixture = JSON.parse(fs.readFileSync(fixturePath, 'utf-8'));

const X = tf.tensor2d(fixture.X as number[][], undefined, 'float32');
const A = SpectralClustering.computeAffinityMatrix(X, {
  affinity: fixture.params.affinity,
  nNeighbors: fixture.params.nNeighbors,
  gamma: fixture.params.gamma,
  nClusters: fixture.params.nClusters,
} as any);

const L = normalised_laplacian(A);

const { eigenvalues } = jacobi_eigen_decomposition(L);
console.log('eigenvalues', eigenvalues.slice(0, 10));

tf.dispose([X, A, L]);
