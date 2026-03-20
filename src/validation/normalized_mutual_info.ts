import { LabelVector } from '../clustering/types';
import { isTensor } from '../utils/tensor-utils';
import { buildContingencyTable, toLabelArray } from './contingency';

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
 * @param labelsTrue - Ground truth class labels
 * @param labelsPred - Predicted cluster labels
 * @param average - Averaging method for normalization (default: 'arithmetic')
 * @returns The Normalized Mutual Information (range: [0, 1])
 * @throws Error if label vectors have different lengths or are empty
 */
export function normalizedMutualInfo(
  labelsTrue: LabelVector,
  labelsPred: LabelVector,
  average: NMIAverage = 'arithmetic',
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

  // Compute entropy H(U) from row sums
  let hTrue = 0;
  for (const a of rowSums) {
    if (a > 0) {
      const p = a / n;
      hTrue -= p * Math.log(p);
    }
  }

  // Compute entropy H(V) from column sums
  let hPred = 0;
  for (const b of colSums) {
    if (b > 0) {
      const p = b / n;
      hPred -= p * Math.log(p);
    }
  }

  // Compute mutual information MI(U, V)
  let mi = 0;
  for (let i = 0; i < table.length; i++) {
    for (let j = 0; j < table[i].length; j++) {
      const nij = table[i][j];
      if (nij > 0) {
        mi += (nij / n) * Math.log((n * nij) / (rowSums[i] * colSums[j]));
      }
    }
  }

  // Compute normalizer based on average method
  let normalizer: number;
  switch (average) {
    case 'arithmetic':
      normalizer = (hTrue + hPred) / 2;
      break;
    case 'geometric':
      normalizer = Math.sqrt(hTrue * hPred);
      break;
    case 'min':
      normalizer = Math.min(hTrue, hPred);
      break;
    case 'max':
      normalizer = Math.max(hTrue, hPred);
      break;
  }

  // When normalizer is 0 and MI is also 0, both clusterings are trivially
  // equivalent (e.g., all points in one cluster). Return 1.0 per sklearn.
  if (normalizer === 0) {
    return mi === 0 ? 1.0 : 0.0;
  }

  return mi / normalizer;
}
