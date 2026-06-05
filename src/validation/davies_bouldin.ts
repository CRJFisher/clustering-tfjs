import * as tf from '../backend/adapter';
import { DataMatrix, LabelVector } from '../clustering/types';
import { validate_labels_length, convert_validation_inputs } from './validate';

/**
 * Computes the Davies-Bouldin score.
 *
 * The Davies-Bouldin index is defined as the average similarity measure
 * of each cluster with its most similar cluster. Lower values indicate
 * better clustering (clusters are more separated).
 *
 * Formula: DB = (1/k) * sum(max_{i≠j}(R_{ij}))
 * where R_{ij} = (s_i + s_j) / d_{ij}
 * - s_i = average distance from points in cluster i to its centroid
 * - d_{ij} = distance between centroids of clusters i and j
 *
 * @param X - Data matrix of shape [n_samples, n_features]
 * @param labels - Cluster labels for each sample
 * @returns The Davies-Bouldin score (lower is better)
 * @throws Error if k <= 1
 */
export function davies_bouldin(X: DataMatrix, labels: LabelVector): number {
  validate_labels_length(X, labels);
  return tf.tidy(() => {
    const { data, label_array } = convert_validation_inputs(X, labels);

    // Get unique labels
    const unique_labels = Array.from(new Set(label_array));
    const k = unique_labels.length;

    // Validate inputs
    if (k <= 1) {
      throw new Error('Davies-Bouldin score requires at least 2 clusters');
    }

    // Compute centroids and intra-cluster dispersions
    const centroids: tf.Tensor1D[] = [];
    const dispersions: number[] = [];

    for (const label of unique_labels) {
      // Get indices for this cluster
      const cluster_indices: number[] = [];
      for (let i = 0; i < label_array.length; i++) {
        if (label_array[i] === label) {
          cluster_indices.push(i);
        }
      }

      const cluster_size = cluster_indices.length;
      if (cluster_size === 0) continue;

      // Extract cluster points
      const cluster_data = tf.gather(data, cluster_indices);

      // Compute centroid
      const centroid = cluster_data.mean(0) as tf.Tensor1D;
      centroids.push(centroid);

      // Compute intra-cluster dispersion (average distance to centroid)
      if (cluster_size > 1) {
        const diff = cluster_data.sub(centroid.reshape([1, -1]));
        const distances = tf.sqrt(diff.square().sum(1));
        const avg_distance = distances.mean().dataSync()[0];
        dispersions.push(avg_distance);
      } else {
        // Single point cluster has zero dispersion
        dispersions.push(0);
      }

    }

    // Compute inter-cluster distances and similarity ratios
    const max_similarities: number[] = [];

    for (let i = 0; i < k; i++) {
      let max_similarity = 0;

      for (let j = 0; j < k; j++) {
        if (i === j) continue;

        // Compute distance between centroids
        const diff = centroids[i].sub(centroids[j]);
        const distance = tf.sqrt(diff.square().sum()).dataSync()[0];

        // Handle coincident centroids (distance === 0)
        if (distance === 0) {
          const numerator = dispersions[i] + dispersions[j];
          if (numerator > 0) {
            // Non-zero dispersion with coincident centroids -> Infinity
            max_similarity = Infinity;
            break;
          }
          // Both dispersions zero with coincident centroids -> skip
          // (matches sklearn nanmax of 0/0 = NaN)
          continue;
        }

        // Compute similarity ratio R_ij = (s_i + s_j) / d_ij
        const similarity = (dispersions[i] + dispersions[j]) / distance;

        if (similarity > max_similarity) {
          max_similarity = similarity;
        }
      }

      max_similarities.push(max_similarity);
    }

    // Compute Davies-Bouldin index as average of maximum similarities
    const db_score = max_similarities.reduce((sum, val) => sum + val, 0) / k;

    return db_score;
  });
}

/**
 * Computes the Davies-Bouldin score with optimized memory usage.
 * This version minimizes tensor allocations and disposals.
 *
 * @param X - Data matrix of shape [n_samples, n_features]
 * @param labels - Cluster labels for each sample
 * @returns The Davies-Bouldin score (lower is better)
 */
export function davies_bouldin_efficient(
  X: DataMatrix,
  labels: LabelVector,
): number {
  validate_labels_length(X, labels);
  const { data, label_array, owns_tensor } = convert_validation_inputs(X, labels);

  // Get unique labels
  const unique_labels = Array.from(new Set(label_array));
  const k = unique_labels.length;

  // Validate
  if (k <= 1) {
    if (owns_tensor) {
      data.dispose();
    }
    throw new Error('Davies-Bouldin score requires at least 2 clusters');
  }

  // Store centroids and dispersions
  const centroid_arrays: number[][] = [];
  const dispersions: number[] = [];

  // Compute centroids and dispersions
  for (const label of unique_labels) {
    const cluster_indices = label_array
      .map((l, i) => (l === label ? i : -1))
      .filter((i) => i >= 0);

    if (cluster_indices.length === 0) continue;

    tf.tidy(() => {
      const cluster_data = tf.gather(data, cluster_indices);
      const centroid = cluster_data.mean(0) as tf.Tensor1D;
      centroid_arrays.push(Array.from(centroid.dataSync()));

      if (cluster_indices.length > 1) {
        const diff = cluster_data.sub(centroid.reshape([1, -1]));
        const distances = tf.sqrt(diff.square().sum(1));
        dispersions.push(distances.mean().dataSync()[0]);
      } else {
        dispersions.push(0);
      }
    });
  }

  // Clean up data tensor if we created it
  if (owns_tensor) {
    data.dispose();
  }

  // Compute Davies-Bouldin index
  let db_sum = 0;

  for (let i = 0; i < k; i++) {
    let max_similarity = 0;

    for (let j = 0; j < k; j++) {
      if (i === j) continue;

      // Compute Euclidean distance between centroids
      let distance = 0;
      for (let d = 0; d < centroid_arrays[i].length; d++) {
        const diff = centroid_arrays[i][d] - centroid_arrays[j][d];
        distance += diff * diff;
      }
      distance = Math.sqrt(distance);

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
