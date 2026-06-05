import * as tf from '../backend/adapter';
import { DataMatrix, LabelVector } from '../clustering/types';
import { validate_labels_length, convert_validation_inputs } from './validate';

/**
 * Computes per-sample silhouette coefficients.
 *
 * The silhouette coefficient for a sample is (b - a) / max(a, b) where:
 * - a is the mean distance between a sample and all other points in the same cluster
 * - b is the mean distance between a sample and all points in the nearest cluster
 *
 * Each score ranges from -1 to +1:
 * - +1: Sample is far from neighboring clusters (well clustered)
 * - 0: Sample is on or very close to the decision boundary
 * - -1: Sample might have been assigned to the wrong cluster
 *
 * @param X - Data matrix of shape [n_samples, n_features]
 * @param labels - Cluster labels for each sample
 * @returns Array of per-sample silhouette scores
 * @throws Error if k <= 1 or labels length doesn't match data rows
 */
export function silhouette_samples(X: DataMatrix, labels: LabelVector): number[] {
  validate_labels_length(X, labels);
  return tf.tidy(() => {
    const { data, label_array } = convert_validation_inputs(X, labels);

    const n = data.shape[0];

    // Get unique labels
    const unique_labels = Array.from(new Set(label_array));
    const k = unique_labels.length;

    // Validate inputs
    if (k <= 1) {
      throw new Error('Silhouette score requires at least 2 clusters');
    }

    // Compute pairwise distances
    // D[i,j] = ||x_i - x_j||^2
    const x_norm = data.square().sum(1).reshape([n, 1]);
    const x_norm_t = x_norm.reshape([1, n]);
    const cross = tf.mat_mul(data, data.transpose());
    const distances = tf.sqrt(
      tf.maximum(tf.scalar(0), x_norm.add(x_norm_t).sub(cross.mul(2))),
    ) as tf.Tensor2D;

    // Compute silhouette for each sample
    const silhouette_values: number[] = [];
    const distances_flat = distances.dataSync() as Float32Array;

    for (let i = 0; i < n; i++) {
      const sample_label = label_array[i];

      // Find indices of samples in same cluster and other clusters
      const same_cluster_indices: number[] = [];
      const other_cluster_indices: Map<number, number[]> = new Map();

      for (let j = 0; j < n; j++) {
        if (i === j) continue;

        if (label_array[j] === sample_label) {
          same_cluster_indices.push(j);
        } else {
          const label = label_array[j];
          if (!other_cluster_indices.has(label)) {
            other_cluster_indices.set(label, []);
          }
          other_cluster_indices.get(label)!.push(j);
        }
      }

      // Compute a(i): mean intra-cluster distance
      let a = 0;
      if (same_cluster_indices.length > 0) {
        for (const j of same_cluster_indices) {
          a += distances_flat[i * n + j];
        }
        a /= same_cluster_indices.length;
      }

      // Compute b(i): mean distance to nearest cluster
      let b = Infinity;
      for (const [_label, indices] of other_cluster_indices) {
        let mean_dist = 0;
        for (const j of indices) {
          mean_dist += distances_flat[i * n + j];
        }
        mean_dist /= indices.length;

        if (mean_dist < b) {
          b = mean_dist;
        }
      }

      // Compute silhouette coefficient
      if (same_cluster_indices.length === 0) {
        // Single point in cluster
        silhouette_values.push(0);
      } else if (a === 0 && b === 0) {
        // All intra- and inter-cluster distances are zero (identical points);
        // sklearn convention: silhouette = 0
        silhouette_values.push(0);
      } else {
        const s = (b - a) / Math.max(a, b);
        silhouette_values.push(s);
      }
    }

    return silhouette_values;
  });
}

/**
 * Computes the Silhouette score (mean of per-sample silhouette coefficients).
 *
 * @param X - Data matrix of shape [n_samples, n_features]
 * @param labels - Cluster labels for each sample
 * @returns The mean silhouette score across all samples
 * @throws Error if k <= 1 or labels length doesn't match data rows
 */
export function silhouette_score(X: DataMatrix, labels: LabelVector): number {
  const samples = silhouette_samples(X, labels);
  return samples.reduce((sum, val) => sum + val, 0) / samples.length;
}

/**
 * Computes the Silhouette score for specific samples (subset).
 * Useful for large datasets where computing all pairwise distances is prohibitive.
 *
 * @param X - Data matrix of shape [n_samples, n_features]
 * @param labels - Cluster labels for each sample
 * @param sample_indices - Indices of samples to compute silhouette for
 * @returns The mean silhouette score for the specified samples
 */
export function silhouette_score_subset(
  X: DataMatrix,
  labels: LabelVector,
  sample_indices: number[],
): number {
  validate_labels_length(X, labels);
  const { data, label_array, owns_tensor } = convert_validation_inputs(X, labels);

  const n = data.shape[0];

  // Get unique labels
  const unique_labels = Array.from(new Set(label_array));
  const k = unique_labels.length;

  // Validate
  if (k <= 1) {
    if (owns_tensor) {
      data.dispose();
    }
    throw new Error('Silhouette score requires at least 2 clusters');
  }

  const silhouette_values: number[] = [];

  // Process each sample in the subset
  for (const i of sample_indices) {
    const sample_label = label_array[i];

    // Get the sample point
    const sample_point = tf.tidy(() => data.gather([i]));

    // Compute distances to all other points
    const distances = tf.tidy(() => {
      const diff = data.sub(sample_point);
      return tf.sqrt(diff.square().sum(1)) as tf.Tensor1D;
    });

    const dist_array = distances.dataSync() as Float32Array;

    // Compute a(i) and b(i)
    let a = 0;
    let a_count = 0;
    const cluster_distances: Map<number, { sum: number; count: number }> =
      new Map();

    for (let j = 0; j < n; j++) {
      if (i === j) continue;

      const dist = dist_array[j];
      const label = label_array[j];

      if (label === sample_label) {
        // Same cluster
        a += dist;
        a_count++;
      } else {
        // Other cluster
        if (!cluster_distances.has(label)) {
          cluster_distances.set(label, { sum: 0, count: 0 });
        }
        const cluster = cluster_distances.get(label)!;
        cluster.sum += dist;
        cluster.count++;
      }
    }

    // Mean intra-cluster distance
    if (a_count > 0) {
      a /= a_count;
    }

    // Find nearest cluster
    let b = Infinity;
    for (const [_label, { sum, count }] of cluster_distances) {
      const mean_dist = sum / count;
      if (mean_dist < b) {
        b = mean_dist;
      }
    }

    // Compute silhouette coefficient
    if (a_count === 0) {
      // Single point in cluster
      silhouette_values.push(0);
    } else if (a === 0 && b === 0) {
      // All intra- and inter-cluster distances are zero (identical points);
      // sklearn convention: silhouette = 0
      silhouette_values.push(0);
    } else {
      const s = (b - a) / Math.max(a, b);
      silhouette_values.push(s);
    }

    // Clean up
    sample_point.dispose();
    distances.dispose();
  }

  // Clean up data tensor if we created it
  if (owns_tensor) {
    data.dispose();
  }

  // Return mean silhouette score
  return (
    silhouette_values.reduce((sum, val) => sum + val, 0) /
    silhouette_values.length
  );
}
