import { LabelVector } from '../clustering/types';
import { build_contingency_table, to_label_array } from './contingency';

/** @throws if label vectors have different lengths or are empty */
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

  const comb2 = (x: number): number => (x * (x - 1)) / 2;

  const n_c2 = comb2(n);

  // If only 0 or 1 sample, partitions are trivially identical
  if (n_c2 === 0) {
    return 1.0;
  }

  let index = 0;
  for (let i = 0; i < table.length; i++) {
    for (let j = 0; j < table[i].length; j++) {
      index += comb2(table[i][j]);
    }
  }

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

  // sklearn: denominator=0 → 1.0 for perfect structural match, else 0.0.
  if (denominator === 0) {
    return index === expected_index ? 1.0 : 0.0;
  }

  return (index - expected_index) / denominator;
}
