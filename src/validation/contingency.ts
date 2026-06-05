import { LabelVector } from '../clustering/types';
import { is_tensor } from '../tensor/tensor_guards';

/**
 * Result of building a contingency table from two label vectors.
 */
export interface ContingencyResult {
  /** Contingency table: table[i][j] = count of (trueClass_i, predClass_j) */
  table: number[][];
  /** Sum across columns for each row */
  row_sums: number[];
  /** Sum across rows for each column */
  col_sums: number[];
  /** Total number of samples */
  n: number;
}

/**
 * Converts a LabelVector (tensor or array) to a plain number[].
 */
export function to_label_array(labels: LabelVector): number[] {
  if (is_tensor(labels)) {
    return Array.from(labels.dataSync() as Float32Array).map((l) =>
      Math.round(l),
    );
  }
  return labels;
}

/**
 * Builds a contingency table from two label vectors.
 *
 * Maps arbitrary integer labels to dense 0-based indices and counts
 * co-occurrences in a 2D matrix.
 *
 * @param labels_true - Ground truth label array
 * @param labels_pred - Predicted label array
 * @returns ContingencyResult with table, row/column sums, and sample count
 */
export function build_contingency_table(
  labels_true: number[],
  labels_pred: number[],
): ContingencyResult {
  const n = labels_true.length;

  // Map unique labels to dense indices
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

  // Build table
  const table: number[][] = Array.from({ length: n_true }, () =>
    new Array(n_pred).fill(0),
  );

  for (let i = 0; i < n; i++) {
    const ti = true_map.get(labels_true[i])!;
    const pi = pred_map.get(labels_pred[i])!;
    table[ti][pi]++;
  }

  // Compute row sums and column sums
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
