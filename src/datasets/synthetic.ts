import * as tf from '../backend/adapter';

export interface MakeBlobsOptions {
  n_samples: number;
  n_features: number;
  centers: number | tf.Tensor2D;
  cluster_std?: number;
  random_state?: number;
}

export interface MakeBlobsResult {
  X: tf.Tensor2D;
  y: number[];
}

export function make_blobs(options: MakeBlobsOptions): MakeBlobsResult {
  const {
    n_samples,
    n_features,
    centers,
    cluster_std = 1.0,
    random_state,
  } = options;

  let centers_tensor: tf.Tensor2D;
  let n_centers: number;

  if (typeof centers === 'number') {
    n_centers = centers;
    // Generate random centers with seed if provided
    centers_tensor = tf.random_uniform(
      [centers, n_features], -10, 10, 'float32',
      random_state,
    );
  } else {
    centers_tensor = centers;
    n_centers = centers_tensor.shape[0];
  }

  // Generate samples
  const samples_per_cluster = Math.floor(n_samples / n_centers);
  const extra_samples = n_samples % n_centers;

  const samples: tf.Tensor2D[] = [];
  const labels: number[] = [];

  for (let i = 0; i < n_centers; i++) {
    const n_samples_cluster = samples_per_cluster + (i < extra_samples ? 1 : 0);

    // Get center for this cluster
    const center = centers_tensor.slice([i, 0], [1, n_features]);

    // Generate samples around this center with a derived seed per cluster
    const cluster_seed = random_state !== undefined ? random_state + i + 1 : undefined;
    const noise = tf.random_normal([n_samples_cluster, n_features], 0, cluster_std, 'float32', cluster_seed);
    const cluster_samples = tf.add(noise, center) as tf.Tensor2D;
    noise.dispose();
    center.dispose();

    samples.push(cluster_samples);
    labels.push(...new Array(n_samples_cluster).fill(i));
  }

  // Concatenate all samples
  const X = tf.concat(samples, 0);

  // Clean up
  if (typeof centers === 'number') {
    centers_tensor.dispose();
  }
  samples.forEach((s) => s.dispose());

  return { X, y: labels };
}
