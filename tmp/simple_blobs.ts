import * as tf from '@tensorflow/tfjs-node';
import { SpectralClustering } from '../src';

function makeBlobs(): tf.Tensor2D {
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
  const model = new SpectralClustering({ nClusters: 2, randomState: 0 });
  const labels = (await model.fitPredict(X)) as number[];
  console.log('labels', labels);
}

run();

