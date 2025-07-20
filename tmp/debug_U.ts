import fs from 'fs';
import * as tf from '@tensorflow/tfjs-node';
import { SpectralClustering } from '../src';
import { normalised_laplacian, smallest_eigenvectors } from '../src/utils/laplacian';

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
const Ufull = smallest_eigenvectors(L, fixture.params.nClusters);
console.log('Ufull shape', Ufull.shape);
const totalCols = Ufull.shape[1];
const trivialCols = totalCols - fixture.params.nClusters;
console.log('trivialCols', trivialCols);
const U = tf.slice(Ufull, [0, trivialCols], [-1, fixture.params.nClusters]);
U.slice([0,0],[10, U.shape[1]]).array().then(arr=>{
  console.log('first 10 rows', arr);
  tf.dispose([X,A,L,Ufull,U]);
});

