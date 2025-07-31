import * as tf from '@tensorflow/tfjs';

export interface MakeBlobsOptions {
  nSamples: number;
  nFeatures: number;
  centers: number | tf.Tensor2D;
  clusterStd?: number;
  randomState?: number;
}

export interface MakeBlobsResult {
  X: tf.Tensor2D;
  y: number[];
}

export function makeBlobs(options: MakeBlobsOptions): MakeBlobsResult {
  const {
    nSamples,
    nFeatures,
    centers,
    clusterStd = 1.0,
    randomState,
  } = options;

  // Set random seed if provided
  if (randomState !== undefined) {
    tf.randomUniform([1], 0, 1, 'float32', randomState);
  }

  let centersTensor: tf.Tensor2D;
  let nCenters: number;

  if (typeof centers === 'number') {
    nCenters = centers;
    // Generate random centers
    centersTensor = tf.randomUniform([centers, nFeatures], -10, 10);
  } else {
    centersTensor = centers;
    nCenters = centersTensor.shape[0];
  }

  // Generate samples
  const samplesPerCluster = Math.floor(nSamples / nCenters);
  const extraSamples = nSamples % nCenters;

  const samples: tf.Tensor2D[] = [];
  const labels: number[] = [];

  for (let i = 0; i < nCenters; i++) {
    const nSamplesCluster = samplesPerCluster + (i < extraSamples ? 1 : 0);

    // Get center for this cluster
    const center = centersTensor.slice([i, 0], [1, nFeatures]);

    // Generate samples around this center
    const noise = tf.randomNormal([nSamplesCluster, nFeatures], 0, clusterStd);
    const clusterSamples = tf.add(noise, center) as tf.Tensor2D;

    samples.push(clusterSamples);
    labels.push(...new Array(nSamplesCluster).fill(i));
  }

  // Concatenate all samples
  const X = tf.concat(samples, 0);

  // Clean up
  if (typeof centers === 'number') {
    centersTensor.dispose();
  }
  samples.forEach((s) => s.dispose());

  return { X, y: labels };
}
