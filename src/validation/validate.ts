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
