import * as tf from '../tf-adapter';
import { DataMatrix, LabelVector } from '../clustering/types';
import { isTensor } from '../utils/tensor-utils';

/**
 * Validates that labels length matches the number of data rows.
 * Should be called at the top of every validation function before
 * any tensor conversion or computation.
 *
 * @param X - Data matrix
 * @param labels - Label vector
 * @throws Error if lengths don't match
 */
export function validateLabelsLength(
  X: DataMatrix,
  labels: LabelVector,
): void {
  const dataRows = isTensor(X) ? X.shape[0] : X.length;
  const labelsLen = isTensor(labels) ? labels.shape[0] : labels.length;

  if (dataRows !== labelsLen) {
    throw new Error(
      `Labels length (${labelsLen}) does not match data rows (${dataRows})`,
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
export function convertValidationInputs(
  X: DataMatrix,
  labels: LabelVector,
): { data: tf.Tensor2D; labelArray: number[]; ownsTensor: boolean } {
  const ownsTensor = !isTensor(X);
  const data = isTensor(X)
    ? (X as tf.Tensor2D)
    : tf.tensor2d(X as number[][]);
  const labelArray = isTensor(labels)
    ? Array.from(labels.dataSync() as Float32Array).map((l) => Math.round(l))
    : (labels as number[]);
  return { data, labelArray, ownsTensor };
}
