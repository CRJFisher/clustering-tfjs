import * as tf from '../backend/adapter';
import type { ClusteringMetric, DataMatrix, LabelVector } from './types';
import { is_tensor } from '../tensor/tensor_guards';
import {
  euclidean_distance,
  manhattan_distance,
  cosine_distance,
} from '../tensor/tensor_ops';

/**
 * Result of {@link select_medoids}: per-cluster medoid sample indices and their
 * distances to the cluster mean. Both arrays are indexed by cluster id and have
 * length `n_clusters`. A cluster with no assigned samples has index `-1` and
 * distance `Infinity`.
 *
 * Medoids are library-defined representatives — scikit-learn exposes no
 * equivalent attribute, so `indices` is covered by behavioural tests rather
 * than reference fixtures.
 */
export interface MedoidResult {
  indices: Int32Array;
  distances: Float32Array;
}

/**
 * Per-point distance to a matching reference point, under the given metric,
 * reusing the shared metric implementations from `tensor_ops`.
 */
function pointwise_distance(
  points: tf.Tensor2D,
  references: tf.Tensor2D,
  metric: ClusteringMetric,
): Float32Array {
  return tf.tidy(() => {
    let dist: tf.Tensor;
    switch (metric) {
      case 'manhattan':
        dist = manhattan_distance(points, references);
        break;
      case 'cosine':
        dist = cosine_distance(points, references);
        break;
      case 'euclidean':
      default:
        dist = euclidean_distance(points, references);
        break;
    }
    return dist.dataSync() as Float32Array;
  });
}

/**
 * For each cluster, finds the index of the sample closest to that cluster's
 * mean under the requested metric — the cluster medoid.
 *
 * Runs in `O(n·d)`: each cluster mean is computed in a single pass and each
 * sample's distance to its own cluster mean in a second pass. No `n×n`
 * pairwise distance matrix is materialised. Distance computation reuses the
 * shared metric implementations in `tensor_ops`.
 *
 * Noise samples (label `-1`) and labels outside `0..n_clusters-1` are ignored.
 * Ties are broken towards the lowest sample index. A cluster with no assigned
 * samples yields index `-1` (no fabricated medoid).
 *
 * @param X Data matrix, shape `n × d`.
 * @param labels Cluster labels, one per sample.
 * @param n_clusters Number of clusters (medoid slots returned).
 * @param metric Distance metric. Defaults to `'euclidean'`.
 */
export async function select_medoids(
  X: DataMatrix,
  labels: LabelVector,
  n_clusters: number,
  metric: ClusteringMetric = 'euclidean',
): Promise<MedoidResult> {
  const data: number[][] = is_tensor(X)
    ? ((await (X as tf.Tensor2D).array()) as number[][])
    : (X as number[][]);
  const label_array: number[] = is_tensor(labels)
    ? Array.from((labels as tf.Tensor1D).dataSync() as Float32Array).map((l) =>
        Math.round(l),
      )
    : (labels as number[]);

  const n = data.length;
  const d = n > 0 ? data[0].length : 0;

  // First pass: per-cluster sums and counts -> means.
  const sums: number[][] = Array.from({ length: n_clusters }, () =>
    new Array<number>(d).fill(0),
  );
  const counts: number[] = new Array<number>(n_clusters).fill(0);
  for (let i = 0; i < n; i++) {
    const l = label_array[i];
    if (l < 0 || l >= n_clusters) continue;
    counts[l]++;
    const row = data[i];
    const s = sums[l];
    for (let f = 0; f < d; f++) s[f] += row[f];
  }
  const means: (number[] | null)[] = sums.map((s, c) =>
    counts[c] > 0 ? s.map((v) => v / counts[c]) : null,
  );

  const indices = new Int32Array(n_clusters).fill(-1);
  const distances = new Float32Array(n_clusters).fill(Number.POSITIVE_INFINITY);

  if (n === 0) {
    return { indices, distances };
  }

  // Build, for every sample, the reference vector = its cluster mean (or the
  // point itself for unassigned samples, which are skipped in the argmin).
  const reference_rows: number[][] = new Array<number[]>(n);
  for (let i = 0; i < n; i++) {
    const l = label_array[i];
    const mean = l >= 0 && l < n_clusters ? means[l] : null;
    reference_rows[i] = mean ?? data[i];
  }

  const point_distances = tf.tidy(() => {
    const points = tf.tensor2d(data, [n, d], 'float32');
    const references = tf.tensor2d(reference_rows, [n, d], 'float32');
    return pointwise_distance(points, references, metric);
  });

  // Second pass: per-cluster argmin (lowest index breaks ties).
  for (let i = 0; i < n; i++) {
    const l = label_array[i];
    if (l < 0 || l >= n_clusters) continue;
    const dist = point_distances[i];
    if (dist < distances[l]) {
      distances[l] = dist;
      indices[l] = i;
    }
  }

  return { indices, distances };
}
