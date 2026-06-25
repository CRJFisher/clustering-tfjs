import * as tf from '../backend/adapter';
import { DataMatrix, LabelVector } from '../clustering/types';
import { is_tensor } from '../tensor/tensor_guards';

export function validate_labels_length(
  X: DataMatrix,
  labels: LabelVector,
): void {
  const data_rows = is_tensor(X) ? X.shape[0] : X.length;
  const labels_len = is_tensor(labels) ? labels.shape[0] : labels.length;

  if (data_rows !== labels_len) {
    throw new Error(
      `Labels length (${labels_len}) does not match data rows (${data_rows})`,
    );
  }
}

/** `owns_tensor` is true when X was an array — the caller must then dispose the returned tensor. */
export function convert_validation_inputs(
  X: DataMatrix,
  labels: LabelVector,
): { data: tf.Tensor2D; label_array: number[]; owns_tensor: boolean } {
  const owns_tensor = !is_tensor(X);
  const data = is_tensor(X)
    ? (X as tf.Tensor2D)
    : tf.tensor2d(X as number[][]);
  const label_array = is_tensor(labels)
    ? Array.from(labels.dataSync() as Float32Array).map((l) => Math.round(l))
    : (labels as number[]);
  return { data, label_array, owns_tensor };
}

/**
 * Density-based estimators emit `-1` for samples that belong to no cluster.
 * Internal-validation metrics measure cohesion and separation of genuine
 * clusters, so noise samples must be removed before any distance or dispersion
 * is computed. The returned `had_noise` flag lets callers distinguish a
 * genuine degenerate input (fewer than two clusters, no noise — an error) from
 * a noise-induced degenerate input (fewer than two clusters remain only after
 * filtering — a defined, non-throwing result).
 */
export function noise_filtered_indices(label_array: number[]): {
  keep: number[];
  had_noise: boolean;
} {
  const keep: number[] = [];
  let had_noise = false;
  for (let i = 0; i < label_array.length; i++) {
    if (label_array[i] === -1) {
      had_noise = true;
    } else {
      keep.push(i);
    }
  }
  return { keep, had_noise };
}
