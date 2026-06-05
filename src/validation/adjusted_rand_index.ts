import { LabelVector } from '../clustering/types';
import { build_contingency_table, to_label_array } from './contingency';

/**
 * Computes the Adjusted Rand Index (ARI) between two clusterings.
 *
 * The ARI is a measure of the similarity between two label assignments,
 * adjusted for chance. It ranges from -1 to 1:
 * - 1: Perfect agreement between the two labelings
 * - 0: Agreement is what would be expected by chance
 * - Negative: Agreement is less than expected by chance
 *
 * The ARI is symmetric and does not depend on the label values themselves,
 * only on the partitioning structure.
 *
 * @param labels_true - Ground truth class labels
 * @param labels_pred - Predicted cluster labels
 * @returns The Adjusted Rand Index (range: [-1, 1])
 * @throws Error if label vectors have different lengths or are empty
 */
export function adjusted_rand_index(
  labels_true: LabelVector,
  labels_pred: LabelVector,
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

  // C(x, 2) = x * (x - 1) / 2
  const comb2 = (x: number): number => (x * (x - 1)) / 2;

  const n_c2 = comb2(n);

  // If only 0 or 1 sample, partitions are trivially identical
  if (n_c2 === 0) {
    return 1.0;
  }

  // sum_ij C(n_ij, 2)
  let index = 0;
  for (let i = 0; i < table.length; i++) {
    for (let j = 0; j < table[i].length; j++) {
      index += comb2(table[i][j]);
    }
  }

  // sum_i C(a_i, 2) and sum_j C(b_j, 2)
  let sum_a = 0;
  for (const a of row_sums) {
    sum_a += comb2(a);
  }

  let sum_b = 0;
  for (const b of col_sums) {
    sum_b += comb2(b);
  }

  const expected_index = (sum_a * sum_b) / n_c2;
  const max_index = (sum_a + sum_b) / 2;

  const denominator = max_index - expected_index;

  // When denominator is 0, check if clusterings are structurally identical.
  // If index === expected_index, all pairs agree (perfect match) -> return 1.0.
  // Otherwise return 0.0 (sklearn convention).
  if (denominator === 0) {
    return index === expected_index ? 1.0 : 0.0;
  }

  return (index - expected_index) / denominator;
}
