import * as tf from '@tensorflow/tfjs-node';
import { normalised_laplacian, smallest_eigenvectors } from '../src/utils/laplacian';
import { compute_rbf_affinity } from '../src/utils/affinity';

function makeBlobs() {
  const pts = [
    [-0.2, 0.1],
    [0.1, -0.1],
    [0.2, 0.2],
    [-0.1, -0.2],
    [5.0, 5.1],
    [5.2, 4.9],
    [4.8, 5.0],
    [5.1, 5.2],
  ];
  return tf.tensor2d(pts, [pts.length, 2], 'float32');
}

async function run() {
  const X = makeBlobs();
  const A = compute_rbf_affinity(X, 1);
  const L = normalised_laplacian(A);
  const U_full = smallest_eigenvectors(L, 2);

  // show eigenvalues as well
  const { jacobi_eigen_decomposition } = require('../src/utils/laplacian');
  const { eigenvalues } = jacobi_eigen_decomposition(L);
  console.log('eigenvalues sorted', eigenvalues.slice(0, 10).map((v: number)=>v.toFixed(6)));
  console.log('U_full shape', U_full.shape);
  const arr = await U_full.array();
  console.log('U_full first rows', arr.map(r => r.map(v => v.toFixed(4))));
  const U = tf.slice(U_full, [0,1], [-1,2]);
  const U_rowNorm = U.norm('euclidean', 1).expandDims(1);
  const U_norm = U.div(U_rowNorm.add(1e-10));
  (await U_norm.array()).forEach((row,i)=> console.log(i,row.map(v=>v.toFixed(4))));
}

run();
