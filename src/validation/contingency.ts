import { LabelVector } from '../clustering/types';
import { isTensor } from '../utils/tensor-utils';

/**
 * Result of building a contingency table from two label vectors.
 */
export interface ContingencyResult {
  /** Contingency table: table[i][j] = count of (trueClass_i, predClass_j) */
  table: number[][];
  /** Sum across columns for each row */
  rowSums: number[];
  /** Sum across rows for each column */
  colSums: number[];
  /** Total number of samples */
  n: number;
}

/**
 * Converts a LabelVector (tensor or array) to a plain number[].
 */
export function toLabelArray(labels: LabelVector): number[] {
  if (isTensor(labels)) {
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
 * @param labelsTrue - Ground truth label array
 * @param labelsPred - Predicted label array
 * @returns ContingencyResult with table, row/column sums, and sample count
 */
export function buildContingencyTable(
  labelsTrue: number[],
  labelsPred: number[],
): ContingencyResult {
  const n = labelsTrue.length;

  // Map unique labels to dense indices
  const trueUnique = Array.from(new Set(labelsTrue));
  const predUnique = Array.from(new Set(labelsPred));

  const trueMap = new Map<number, number>();
  for (let i = 0; i < trueUnique.length; i++) {
    trueMap.set(trueUnique[i], i);
  }

  const predMap = new Map<number, number>();
  for (let i = 0; i < predUnique.length; i++) {
    predMap.set(predUnique[i], i);
  }

  const nTrue = trueUnique.length;
  const nPred = predUnique.length;

  // Build table
  const table: number[][] = Array.from({ length: nTrue }, () =>
    new Array(nPred).fill(0),
  );

  for (let i = 0; i < n; i++) {
    const ti = trueMap.get(labelsTrue[i])!;
    const pi = predMap.get(labelsPred[i])!;
    table[ti][pi]++;
  }

  // Compute row sums and column sums
  const rowSums = new Array(nTrue).fill(0);
  const colSums = new Array(nPred).fill(0);

  for (let i = 0; i < nTrue; i++) {
    for (let j = 0; j < nPred; j++) {
      rowSums[i] += table[i][j];
      colSums[j] += table[i][j];
    }
  }

  return { table, rowSums, colSums, n };
}
