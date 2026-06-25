import * as tf from '../backend/adapter';
import { DataMatrix, LabelVector } from '../clustering/types';
import {
  validate_labels_length,
  convert_validation_inputs,
  noise_filtered_indices,
} from './validate';
import { pairwise_distance_matrix } from '../distance/pairwise_distance';
import type { ValidationMetric } from './silhouette';

/**
 * For cosine metric: L2-normalises the mean to a unit vector. Semantic requirement
 * even though cluster_dispersion and pairwise_distance_matrix normalise internally
 * (making the score invariant). The practical benefit is in davies_bouldin_efficient
 * where the centroid is serialised to a JS array: a unit vector avoids directional
 * precision loss for near-zero-magnitude means. When the mean is exactly [0,…,0]
 * (perfectly-antipodal cluster) the direction is undefined and the centroid stays zero.
 */
function cluster_centroid(
  cluster_data: tf.Tensor2D,
  metric: ValidationMetric,
): tf.Tensor1D {
  const mean = cluster_data.mean(0) as tf.Tensor1D;
  if (metric !== 'cosine') return mean;
  const norm = mean.square().sum().sqrt().add(tf.scalar(1e-8));
  return mean.div(norm) as tf.Tensor1D;
}

function cluster_dispersion(
  cluster_data: tf.Tensor2D,
  centroid: tf.Tensor1D,
  metric: ValidationMetric,
): number {
  return tf.tidy(() => {
    if (metric === 'cosine') {
      const eps = tf.scalar(1e-8);
      const dot = cluster_data.mul(centroid).sum(1);
      const data_norm = cluster_data.square().sum(1).sqrt();
      const centroid_norm = centroid.square().sum().sqrt();
      const sim = dot.div(data_norm.mul(centroid_norm).add(eps));
      return tf.scalar(1).sub(sim).mean().dataSync()[0];
    }
    const diff = cluster_data.sub(centroid.reshape([1, -1]));
    return tf.sqrt(diff.square().sum(1)).mean().dataSync()[0];
  });
}

/**
 * Formula: DB = (1/k) * sum(max_{i≠j}(R_{ij}))
 * where R_{ij} = (s_i + s_j) / d_{ij},
 * s_i = mean distance of cluster i's points to its centroid,
 * d_{ij} = distance between centroids of i and j.
 *
 * Noise (`-1`) samples are excluded before centroid/dispersion computation.
 * When noise leaves fewer than two clusters the score is a defined `0` rather than an error.
 *
 * @throws Error if there are fewer than 2 clusters and no noise was present
 */
export function davies_bouldin(
  X: DataMatrix,
  labels: LabelVector,
  metric: ValidationMetric = 'euclidean',
): number {
  validate_labels_length(X, labels);
  return tf.tidy(() => {
    const { data, label_array } = convert_validation_inputs(X, labels);

    const { keep, had_noise } = noise_filtered_indices(label_array);
    const work_data = had_noise ? (data.gather(keep) as tf.Tensor2D) : data;
    const work_labels = had_noise
      ? keep.map((i) => label_array[i])
      : label_array;

    const unique_labels = Array.from(new Set(work_labels));
    const k = unique_labels.length;

    if (k <= 1) {
      // Noise-induced degenerate case is defined and non-throwing.
      if (had_noise) {
        return 0;
      }
      throw new Error('Davies-Bouldin score requires at least 2 clusters');
    }

    const centroids: tf.Tensor1D[] = [];
    const dispersions: number[] = [];

    for (const label of unique_labels) {
      const cluster_indices: number[] = [];
      for (let i = 0; i < work_labels.length; i++) {
        if (work_labels[i] === label) {
          cluster_indices.push(i);
        }
      }

      // unique_labels is derived from work_labels, so every label has members.
      const cluster_size = cluster_indices.length;
      const cluster_data = tf.gather(work_data, cluster_indices) as tf.Tensor2D;
      const centroid = cluster_centroid(cluster_data, metric);
      centroids.push(centroid);
      dispersions.push(cluster_size > 1 ? cluster_dispersion(cluster_data, centroid, metric) : 0);
    }

    const centroid_matrix = tf.stack(centroids) as tf.Tensor2D;
    const centroid_distances = pairwise_distance_matrix(
      centroid_matrix,
      metric,
    ).arraySync() as number[][];

    const max_similarities: number[] = [];

    for (let i = 0; i < k; i++) {
      let max_similarity = 0;

      for (let j = 0; j < k; j++) {
        if (i === j) continue;

        const distance = centroid_distances[i][j];

        if (distance === 0) {
          const numerator = dispersions[i] + dispersions[j];
          if (numerator > 0) {
            // Non-zero dispersion with coincident centroids -> Infinity.
            max_similarity = Infinity;
            break;
          }
          // Both dispersions zero with coincident centroids -> skip
          // (matches sklearn nanmax of 0/0 = NaN).
          continue;
        }

        const similarity = (dispersions[i] + dispersions[j]) / distance;
        if (similarity > max_similarity) {
          max_similarity = similarity;
        }
      }

      max_similarities.push(max_similarity);
    }

    const db_score = max_similarities.reduce((sum, val) => sum + val, 0) / k;

    return db_score;
  });
}

/** Manages tensor lifecycle manually (no tf.tidy wrapper) to bound peak memory for large datasets. */
export function davies_bouldin_efficient(
  X: DataMatrix,
  labels: LabelVector,
  metric: ValidationMetric = 'euclidean',
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

  const unique_labels = Array.from(new Set(work_labels));
  const k = unique_labels.length;

  if (k <= 1) {
    dispose_work();
    // Noise-induced degenerate case is defined and non-throwing.
    if (had_noise) {
      return 0;
    }
    throw new Error('Davies-Bouldin score requires at least 2 clusters');
  }

  const centroid_arrays: number[][] = [];
  const dispersions: number[] = [];

  for (const label of unique_labels) {
    const cluster_indices = work_labels
      .map((l, i) => (l === label ? i : -1))
      .filter((i) => i >= 0);

    tf.tidy(() => {
      const cluster_data = tf.gather(work_data, cluster_indices) as tf.Tensor2D;
      const centroid = cluster_centroid(cluster_data, metric);
      centroid_arrays.push(Array.from(centroid.dataSync()));

      if (cluster_indices.length > 1) {
        dispersions.push(cluster_dispersion(cluster_data, centroid, metric));
      } else {
        dispersions.push(0);
      }
    });
  }

  const centroid_distances = tf.tidy(() => {
    const centroid_matrix = tf.tensor2d(centroid_arrays);
    return pairwise_distance_matrix(centroid_matrix, metric).arraySync() as number[][];
  });

  dispose_work();

  let db_sum = 0;

  for (let i = 0; i < k; i++) {
    let max_similarity = 0;

    for (let j = 0; j < k; j++) {
      if (i === j) continue;

      const distance = centroid_distances[i][j];

      if (distance === 0) {
        const numerator = dispersions[i] + dispersions[j];
        if (numerator > 0) {
          max_similarity = Infinity;
          break;
        }
        // Both dispersions zero with coincident centroids -> skip
        // (matches sklearn nanmax of 0/0 = NaN)
        continue;
      }

      const similarity = (dispersions[i] + dispersions[j]) / distance;
      if (similarity > max_similarity) {
        max_similarity = similarity;
      }
    }

    db_sum += max_similarity;
  }

  return db_sum / k;
}
