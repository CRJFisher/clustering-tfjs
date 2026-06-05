import * as tf from '../backend/adapter';
import { DataMatrix, LabelVector } from '../clustering/types';
import { is_tensor } from '../tensor/tensor_guards';

/**
 * Validates that labels length matches the number of data rows.
 * Should be called at the top of every validation function before
 * any tensor conversion or computation.
 *
 * @param X - Data matrix
 * @param labels - Label vector
 * @throws Error if lengths don't match
 */
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

/**
 * Converts DataMatrix and LabelVector inputs to their working forms.
 * Returns the tensor and a plain number[] of integer labels.
 *
 * @param X - Data matrix (tensor or array)
 * @param labels - Label vector (tensor or array)
 * @returns Object with data tensor, label array, and whether the tensor is owned (and should be disposed by caller)
 */
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
