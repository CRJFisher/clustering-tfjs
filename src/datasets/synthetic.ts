import * as tf from '../tf-adapter';

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

  let centersTensor: tf.Tensor2D;
  let nCenters: number;

  if (typeof centers === 'number') {
    nCenters = centers;
    // Generate random centers with seed if provided
    centersTensor = tf.randomUniform(
      [centers, nFeatures], -10, 10, 'float32',
      randomState,
    );
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

    // Generate samples around this center with a derived seed per cluster
    const clusterSeed = randomState !== undefined ? randomState + i + 1 : undefined;
    const noise = tf.randomNormal([nSamplesCluster, nFeatures], 0, clusterStd, 'float32', clusterSeed);
    const clusterSamples = tf.add(noise, center) as tf.Tensor2D;
    noise.dispose();
    center.dispose();

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
