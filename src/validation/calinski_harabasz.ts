import * as tf from '../backend/adapter';
import { DataMatrix, LabelVector } from '../clustering/types';
import {
  validate_labels_length,
  convert_validation_inputs,
  noise_filtered_indices,
} from './validate';

/**
 * Computes the Calinski-Harabasz score (also known as Variance Ratio Criterion).
 *
 * The score is defined as the ratio of the between-cluster dispersion to the
 * within-cluster dispersion. Higher values indicate better-defined clusters.
 *
 * Formula: CH = (BSS / (k - 1)) / (WSS / (n - k))
 * where:
 * - BSS = between-cluster sum of squares
 * - WSS = within-cluster sum of squares
 * - k = number of clusters
 * - n = number of samples
 *
 * @param X - Data matrix of shape [n_samples, n_features]
 * @param labels - Cluster labels for each sample
 * @returns The Calinski-Harabasz score (higher is better)
 * @throws Error if k <= 1 or k >= n_samples
 */
export function calinski_harabasz(X: DataMatrix, labels: LabelVector): number {
  validate_labels_length(X, labels);
  return tf.tidy(() => {
    const { data, label_array } = convert_validation_inputs(X, labels);

    // Exclude noise (-1) samples before computing any sum-of-squares.
    const { keep, had_noise } = noise_filtered_indices(label_array);
    const work_data = had_noise ? (data.gather(keep) as tf.Tensor2D) : data;
    const work_labels = had_noise
      ? keep.map((i) => label_array[i])
      : label_array;

    const n = work_data.shape[0];

    // Get unique labels and count
    const unique_labels = Array.from(new Set(work_labels));
    const k = unique_labels.length;

    // Validate inputs
    if (k <= 1) {
      // Noise-induced degenerate case is defined and non-throwing.
      if (had_noise) {
        return 0;
      }
      throw new Error('Calinski-Harabasz score requires at least 2 clusters');
    }
    if (k >= n) {
      if (had_noise) {
        return 0;
      }
      throw new Error('Number of clusters must be less than number of samples');
    }

    // Compute global centroid
    const global_centroid = work_data.mean(0) as tf.Tensor1D;

    // Initialize accumulators
    let within_cluster_ss = 0;
    let between_cluster_ss = 0;

    // Process each cluster
    for (const label of unique_labels) {
      // Get indices for this cluster
      const cluster_indices: number[] = [];
      for (let i = 0; i < work_labels.length; i++) {
        if (work_labels[i] === label) {
          cluster_indices.push(i);
        }
      }

      const cluster_size = cluster_indices.length;

      // Extract cluster points
      const cluster_data = tf.gather(work_data, cluster_indices);

      // Compute cluster centroid
      const cluster_centroid = cluster_data.mean(0) as tf.Tensor1D;

      // Within-cluster sum of squares
      const diff = cluster_data.sub(cluster_centroid.reshape([1, -1]));
      const squared_diff = diff.square();
      within_cluster_ss += squared_diff.sum().dataSync()[0];

      // Between-cluster sum of squares
      const centroid_diff = cluster_centroid.sub(global_centroid);
      const centroid_diff_squared = centroid_diff.square().sum();
      between_cluster_ss += cluster_size * centroid_diff_squared.dataSync()[0];

    }

    // Compute Calinski-Harabasz score
    const score = between_cluster_ss / (k - 1) / (within_cluster_ss / (n - k));

    return score;
  });
}

/**
 * Computes the Calinski-Harabasz score in a memory-efficient manner for large datasets.
 * This version processes clusters sequentially to minimize memory usage.
 *
 * @param X - Data matrix of shape [n_samples, n_features]
 * @param labels - Cluster labels for each sample
 * @returns The Calinski-Harabasz score
 */
export function calinski_harabasz_efficient(
  X: DataMatrix,
  labels: LabelVector,
): number {
  validate_labels_length(X, labels);
  const { data, label_array, owns_tensor } = convert_validation_inputs(X, labels);

  // Exclude noise (-1) samples before computing any sum-of-squares.
  const { keep, had_noise } = noise_filtered_indices(label_array);
  const work_data = had_noise ? (tf.gather(data, keep) as tf.Tensor2D) : data;
  const work_labels = had_noise ? keep.map((i) => label_array[i]) : label_array;
  const dispose_work = (): void => {
    if (had_noise) work_data.dispose();
    if (owns_tensor) data.dispose();
  };

  const n = work_data.shape[0];

  // Get unique labels
  const unique_labels = Array.from(new Set(work_labels));
  const k = unique_labels.length;

  // Validate
  if (k <= 1) {
    dispose_work();
    // Noise-induced degenerate case is defined and non-throwing.
    if (had_noise) {
      return 0;
    }
    throw new Error('Calinski-Harabasz score requires at least 2 clusters');
  }
  if (k >= n) {
    dispose_work();
    if (had_noise) {
      return 0;
    }
    throw new Error('Number of clusters must be less than number of samples');
  }

  // Compute global centroid
  const global_centroid = tf.tidy(() => work_data.mean(0) as tf.Tensor1D);

  let within_cluster_ss = 0;
  let between_cluster_ss = 0;

  // Process each cluster
  for (const label of unique_labels) {
    tf.tidy(() => {
      // Get cluster indices
      const cluster_indices = work_labels
        .map((l, i) => (l === label ? i : -1))
        .filter((i) => i >= 0);

      const cluster_size = cluster_indices.length;

      // Extract cluster data
      const cluster_data = tf.gather(work_data, cluster_indices);
      const cluster_centroid = cluster_data.mean(0) as tf.Tensor1D;

      // Within-cluster SS
      const diff = cluster_data.sub(cluster_centroid.reshape([1, -1]));
      within_cluster_ss += diff.square().sum().dataSync()[0];

      // Between-cluster SS
      const centroid_diff = cluster_centroid.sub(global_centroid);
      between_cluster_ss +=
        cluster_size * centroid_diff.square().sum().dataSync()[0];
    });
  }

  // Clean up
  global_centroid.dispose();
  dispose_work();

  // Compute score
  return between_cluster_ss / (k - 1) / (within_cluster_ss / (n - k));
}
