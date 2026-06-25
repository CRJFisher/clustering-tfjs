import * as tf from '../backend/adapter';
import { DataMatrix, LabelVector } from '../clustering/types';
import { convert_validation_inputs } from '../validation/validate';

/** WSS = Σ ||x_i − centroid_c||²; called inertia in sklearn. */
export function compute_wss(X: DataMatrix, labels: LabelVector): number {
  const { data, label_array, owns_tensor } = convert_validation_inputs(X, labels);

  const unique_labels = Array.from(new Set(label_array));
  let wss = 0;

  for (const label of unique_labels) {
    tf.tidy(() => {
      const cluster_indices = label_array
        .map((l, i) => (l === label ? i : -1))
        .filter((i) => i >= 0);

      if (cluster_indices.length === 0) return;

      const cluster_data = tf.gather(data, cluster_indices);
      const centroid = cluster_data.mean(0) as tf.Tensor1D;
      const diff = cluster_data.sub(centroid.reshape([1, -1]));
      wss += diff.square().sum().dataSync()[0];
    });
  }

  if (owns_tensor) {
    data.dispose();
  }

  return wss;
}
