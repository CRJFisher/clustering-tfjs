import { LabelVector } from '../clustering/types';
import { build_contingency_table, to_label_array } from './contingency';

/**
 * Averaging method for NMI normalization.
 * - 'arithmetic': (H(U) + H(V)) / 2
 * - 'geometric': sqrt(H(U) * H(V))
 * - 'min': min(H(U), H(V))
 * - 'max': max(H(U), H(V))
 */
export type NMIAverage = 'arithmetic' | 'geometric' | 'min' | 'max';

/**
 * Computes the Normalized Mutual Information (NMI) between two clusterings.
 *
 * NMI is a normalization of the mutual information score to scale the
 * result to [0, 1]. It measures the agreement between two label assignments:
 * - 1: Perfect agreement
 * - 0: No mutual information (independent labelings)
 *
 * @param labels_true - Ground truth class labels
 * @param labels_pred - Predicted cluster labels
 * @param average - Averaging method for normalization (default: 'arithmetic')
 * @returns The Normalized Mutual Information (range: [0, 1])
 * @throws Error if label vectors have different lengths or are empty
 */
export function normalized_mutual_info(
  labels_true: LabelVector,
  labels_pred: LabelVector,
  average: NMIAverage = 'arithmetic',
): number {
  const true_arr = to_label_array(labels_true);
  const pred_arr = to_label_array(labels_pred);

  if (true_arr.length === 0) {
    throw new Error('Label vectors must not be empty');
  }

  const true_len = true_arr.length;
  const pred_len = pred_arr.length;

  if (true_len !== pred_len) {
    throw new Error(
      `Label vectors must have the same length (got ${true_len} and ${pred_len})`,
    );
  }

  const { table, row_sums, col_sums, n } = build_contingency_table(true_arr, pred_arr);

  let h_true = 0;
  for (const a of row_sums) {
    if (a > 0) {
      const p = a / n;
      h_true -= p * Math.log(p);
    }
  }

  let h_pred = 0;
  for (const b of col_sums) {
    if (b > 0) {
      const p = b / n;
      h_pred -= p * Math.log(p);
    }
  }

  let mi = 0;
  for (let i = 0; i < table.length; i++) {
    for (let j = 0; j < table[i].length; j++) {
      const nij = table[i][j];
      if (nij > 0) {
        mi += (nij / n) * Math.log((n * nij) / (row_sums[i] * col_sums[j]));
      }
    }
  }

  let normalizer: number;
  switch (average) {
    case 'arithmetic':
      normalizer = (h_true + h_pred) / 2;
      break;
    case 'geometric':
      normalizer = Math.sqrt(h_true * h_pred);
      break;
    case 'min':
      normalizer = Math.min(h_true, h_pred);
      break;
    case 'max':
      normalizer = Math.max(h_true, h_pred);
      break;
  }

  // When normalizer is 0 and MI is also 0, both clusterings are trivially
  // equivalent (e.g., all points in one cluster). Return 1.0 per sklearn.
  if (normalizer === 0) {
    return mi === 0 ? 1.0 : 0.0;
  }

  return mi / normalizer;
}
