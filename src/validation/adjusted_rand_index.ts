import { LabelVector } from '../clustering/types';
import { isTensor } from '../utils/tensor-utils';
import { buildContingencyTable, toLabelArray } from './contingency';

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
 * @param labelsTrue - Ground truth class labels
 * @param labelsPred - Predicted cluster labels
 * @returns The Adjusted Rand Index (range: [-1, 1])
 * @throws Error if label vectors have different lengths or are empty
 */
export function adjustedRandIndex(
  labelsTrue: LabelVector,
  labelsPred: LabelVector,
): number {
  const trueArr = toLabelArray(labelsTrue);
  const predArr = toLabelArray(labelsPred);

  if (trueArr.length === 0) {
    throw new Error('Label vectors must not be empty');
  }

  const trueLen = isTensor(labelsTrue) ? labelsTrue.shape[0] : trueArr.length;
  const predLen = isTensor(labelsPred) ? labelsPred.shape[0] : predArr.length;

  if (trueLen !== predLen) {
    throw new Error(
      `Label vectors must have the same length (got ${trueLen} and ${predLen})`,
    );
  }

  const { table, rowSums, colSums, n } = buildContingencyTable(trueArr, predArr);

  // C(x, 2) = x * (x - 1) / 2
  const comb2 = (x: number): number => (x * (x - 1)) / 2;

  const nC2 = comb2(n);

  // If only 0 or 1 sample, ARI is 0
  if (nC2 === 0) {
    return 0;
  }

  // sum_ij C(n_ij, 2)
  let index = 0;
  for (let i = 0; i < table.length; i++) {
    for (let j = 0; j < table[i].length; j++) {
      index += comb2(table[i][j]);
    }
  }

  // sum_i C(a_i, 2) and sum_j C(b_j, 2)
  let sumA = 0;
  for (const a of rowSums) {
    sumA += comb2(a);
  }

  let sumB = 0;
  for (const b of colSums) {
    sumB += comb2(b);
  }

  const expectedIndex = (sumA * sumB) / nC2;
  const maxIndex = (sumA + sumB) / 2;

  const denominator = maxIndex - expectedIndex;

  // When denominator is 0 (trivial clustering), return 0 per sklearn convention
  if (denominator === 0) {
    return 0;
  }

  return (index - expectedIndex) / denominator;
}
