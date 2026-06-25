import * as tf from '../backend/adapter';
import { DataMatrix, LabelVector } from '../clustering/types';
import {
  validate_labels_length,
  convert_validation_inputs,
  noise_filtered_indices,
} from './validate';

/**
 * Formula: CH = (BSS / (k - 1)) / (WSS / (n - k))
 * where BSS = between-cluster sum of squares, WSS = within-cluster sum of squares.
 *
 * Noise (`-1`) samples are excluded before computing. When noise leaves fewer than
 * two clusters the score is a defined `0` rather than an error. Variance-based,
 * so no `metric` parameter.
 *
 * @throws Error if there are fewer than 2 clusters and no noise was present
 */
export function calinski_harabasz(X: DataMatrix, labels: LabelVector): number {
  validate_labels_length(X, labels);
  return tf.tidy(() => {
    const { data, label_array } = convert_validation_inputs(X, labels);

    const { keep, had_noise } = noise_filtered_indices(label_array);
    const work_data = had_noise ? (data.gather(keep) as tf.Tensor2D) : data;
    const work_labels = had_noise
      ? keep.map((i) => label_array[i])
      : label_array;

    const n = work_data.shape[0];

    const unique_labels = Array.from(new Set(work_labels));
    const k = unique_labels.length;

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

    const global_centroid = work_data.mean(0) as tf.Tensor1D;

    let within_cluster_ss = 0;
    let between_cluster_ss = 0;

    for (const label of unique_labels) {
      const cluster_indices: number[] = [];
      for (let i = 0; i < work_labels.length; i++) {
        if (work_labels[i] === label) {
          cluster_indices.push(i);
        }
      }

      const cluster_size = cluster_indices.length;
      const cluster_data = tf.gather(work_data, cluster_indices);
      const cluster_centroid = cluster_data.mean(0) as tf.Tensor1D;

      const diff = cluster_data.sub(cluster_centroid.reshape([1, -1]));
      within_cluster_ss += diff.square().sum().dataSync()[0];

      const centroid_diff = cluster_centroid.sub(global_centroid);
      between_cluster_ss += cluster_size * centroid_diff.square().sum().dataSync()[0];
    }

    const score = between_cluster_ss / (k - 1) / (within_cluster_ss / (n - k));

    return score;
  });
}

/** Manages tensor lifecycle manually (no tf.tidy wrapper) to bound peak memory for large datasets. */
export function calinski_harabasz_efficient(
  X: DataMatrix,
  labels: LabelVector,
): number {
  validate_labels_length(X, labels);
  const { data, label_array, owns_tensor } = convert_validation_inputs(X, labels);

  const { keep, had_noise } = noise_filtered_indices(label_array);
  const work_data = had_noise ? (tf.gather(data, keep) as tf.Tensor2D) : data;
  const work_labels = had_noise ? keep.map((i) => label_array[i]) : label_array;
  const dispose_work = (): void => {
    if (had_noise) work_data.dispose();
    if (owns_tensor) data.dispose();
  };

  const n = work_data.shape[0];

  const unique_labels = Array.from(new Set(work_labels));
  const k = unique_labels.length;

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

  const global_centroid = tf.tidy(() => work_data.mean(0) as tf.Tensor1D);

  let within_cluster_ss = 0;
  let between_cluster_ss = 0;

  for (const label of unique_labels) {
    tf.tidy(() => {
      const cluster_indices = work_labels
        .map((l, i) => (l === label ? i : -1))
        .filter((i) => i >= 0);

      const cluster_size = cluster_indices.length;
      const cluster_data = tf.gather(work_data, cluster_indices);
      const cluster_centroid = cluster_data.mean(0) as tf.Tensor1D;

      const diff = cluster_data.sub(cluster_centroid.reshape([1, -1]));
      within_cluster_ss += diff.square().sum().dataSync()[0];

      const centroid_diff = cluster_centroid.sub(global_centroid);
      between_cluster_ss +=
        cluster_size * centroid_diff.square().sum().dataSync()[0];
    });
  }

  global_centroid.dispose();
  dispose_work();

  return between_cluster_ss / (k - 1) / (within_cluster_ss / (n - k));
}
