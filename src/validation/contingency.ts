import { LabelVector } from '../clustering/types';
import { is_tensor } from '../tensor/tensor_guards';

export interface ContingencyResult {
  table: number[][];
  row_sums: number[];
  col_sums: number[];
  n: number;
}

export function to_label_array(labels: LabelVector): number[] {
  if (is_tensor(labels)) {
    // float32 storage can shift integer labels by a tiny epsilon; round to recover them.
    return Array.from(labels.dataSync() as Float32Array).map((l) =>
      Math.round(l),
    );
  }
  return labels;
}

/** Maps arbitrary integer labels to dense 0-based indices before counting co-occurrences. */
export function build_contingency_table(
  labels_true: number[],
  labels_pred: number[],
): ContingencyResult {
  const n = labels_true.length;

  const true_unique = Array.from(new Set(labels_true));
  const pred_unique = Array.from(new Set(labels_pred));

  const true_map = new Map<number, number>();
  for (let i = 0; i < true_unique.length; i++) {
    true_map.set(true_unique[i], i);
  }

  const pred_map = new Map<number, number>();
  for (let i = 0; i < pred_unique.length; i++) {
    pred_map.set(pred_unique[i], i);
  }

  const n_true = true_unique.length;
  const n_pred = pred_unique.length;

  const table: number[][] = Array.from({ length: n_true }, () =>
    new Array(n_pred).fill(0),
  );

  for (let i = 0; i < n; i++) {
    const ti = true_map.get(labels_true[i])!;
    const pi = pred_map.get(labels_pred[i])!;
    table[ti][pi]++;
  }

  const row_sums = new Array(n_true).fill(0);
  const col_sums = new Array(n_pred).fill(0);

  for (let i = 0; i < n_true; i++) {
    for (let j = 0; j < n_pred; j++) {
      row_sums[i] += table[i][j];
      col_sums[j] += table[i][j];
    }
  }

  return { table, row_sums, col_sums, n };
}
