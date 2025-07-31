/**
 * Utility helpers implementing Lance–Williams update formulas for the most
 * common hierarchical clustering linkage criteria.
 *
 * The functions work on an explicit distance matrix represented as a
 * JavaScript number[][] (2-D array) to avoid pulling TensorFlow into the
 * critical inner loop. This keeps the implementation lightweight, easy to
 * test and free of any GC pressure from temporary tensors.
 *
 * The matrix *must* satisfy the following conditions:
 *   • Square (n × n)
 *   • Symmetric:   D[i][j] === D[j][i]
 *   • Zero diagonal: D[i][i] === 0 for all i
 *
 * When two clusters "i" and "j" are merged into a new cluster "t" the
 * distance matrix needs to be updated by computing the distance between "t"
 * and each remaining cluster "k" according to the chosen linkage criterion.
 *
 * The Lance–Williams recurrence expresses the updated distance D(t,k) as a
 * linear combination of the previous distances:
 *
 *          D(t,k) =  α_i  · D(i,k)
 *                   + α_j  · D(j,k)
 *                   +  β   · D(i,j)
 *                   + γ · | D(i,k) − D(j,k) |
 *
 * For the four linkage strategies implemented in this module the parameters
 * are:
 *
 *   • Single   : α_i = α_j = 0.5,  β = 0,        γ = -0.5
 *   • Complete : α_i = α_j = 0.5,  β = 0,        γ = +0.5
 *   • Average  : α_i = n_i / (n_i + n_j),
 *                α_j = n_j / (n_i + n_j),  β = 0, γ = 0
 *   • Ward     : α_i = (n_i + n_k) / (n_i + n_j + n_k),
 *                α_j = (n_j + n_k) / (n_i + n_j + n_k),
 *                β   = -n_k / (n_i + n_j + n_k),        γ = 0
 *
 * However, for Single/Complete/Average it is considerably cheaper and more
 * intuitive to compute the updated distance directly (min, max, weighted
 * mean) instead of evaluating the general formula above. Ward linkage on the
 * other hand is implemented using the Lance-Williams coefficients because it
 * requires them for numerical stability.
 */

export type LinkageCriterion = 'single' | 'complete' | 'average' | 'ward';

/** Distance matrix represented as a square 2-D number array. */
export type DistanceMatrix = number[][];

/**
 * Updates the distance matrix after merging clusters `i` and `j` into a new
 * cluster. The function mutates the matrix *in place* and returns it for
 * convenience.
 *
 * The row/column with the larger index is removed to keep indices stable for
 * the caller (mirroring the typical implementation in hierarchical
 * clustering libraries). After the merge, entry `i` of `clusterSizes` is
 * overwritten by the new cluster size while entry `j` is removed — keeping
 * the length of the array consistent with the contracted distance matrix.
 *
 * Parameters
 * ----------
 * D             – symmetric distance matrix (will be mutated)
 * clusterSizes  – array holding the size (number of original samples) of each
 *                 current cluster. Must have the same length as D.
 * i, j          – indices of the clusters to be merged (i < j).
 * linkage       – linkage criterion used for the update.
 */
export function update_distance_matrix(
  D: DistanceMatrix,
  clusterSizes: number[],
  i: number,
  j: number,
  linkage: LinkageCriterion,
): DistanceMatrix {
  if (i === j) {
    throw new Error('Cannot merge the same cluster (i === j).');
  }
  if (i > j) {
    // Ensure i < j to simplify row/col removal logic.
    [i, j] = [j, i];
  }

  const nClusters = D.length;
  if (clusterSizes.length !== nClusters) {
    throw new Error('clusterSizes length must match D dimensions.');
  }

  const sizeI = clusterSizes[i];
  const sizeJ = clusterSizes[j];
  const newSize = sizeI + sizeJ;

  // Temporary array to hold the new distances from the merged cluster to all
  // other clusters prior to shrinking the matrix.
  const newDistances: number[] = new Array(nClusters).fill(0);

  for (let k = 0; k < nClusters; k++) {
    if (k === i || k === j) continue; // skip self distances

    const dik = D[i][k];
    const djk = D[j][k];

    let dtk: number;
    switch (linkage) {
      case 'single':
        dtk = Math.min(dik, djk);
        break;
      case 'complete':
        dtk = Math.max(dik, djk);
        break;
      case 'average':
        dtk = (sizeI * dik + sizeJ * djk) / newSize;
        break;
      case 'ward': {
        const sizeK = clusterSizes[k];
        const dij = D[i][j];
        const numerator =
          (sizeI + sizeK) * dik * dik +
          (sizeJ + sizeK) * djk * djk -
          sizeK * dij * dij;
        const denom = newSize + sizeK;
        dtk = Math.sqrt(Math.max(numerator / denom, 0));
        break;
      }
      default:
        // Exhaustiveness guard – should never happen.
        // eslint-disable-next-line @typescript-eslint/restrict-template-expressions
        throw new Error(`Unsupported linkage '${(linkage as string) ?? ''}'.`);
    }

    newDistances[k] = dtk;
  }

  /* ------------------------------------------------------------------ */
  /*          Replace row/column i with the new distances & shrink        */
  /* ------------------------------------------------------------------ */

  // Copy the new distances into both row and column i to preserve symmetry.
  for (let k = 0; k < nClusters; k++) {
    if (k === i) {
      D[i][k] = 0;
    } else if (k === j) {
      // Will be removed later.
      continue;
    } else {
      D[i][k] = D[k][i] = newDistances[k];
    }
  }

  // Remove row j.
  D.splice(j, 1);
  // Remove column j.
  for (let row = 0; row < D.length; row++) {
    D[row].splice(j, 1);
  }

  // Update cluster sizes: overwrite entry i with merged size, remove j, append.
  clusterSizes[i] = newSize;
  clusterSizes.splice(j, 1);

  return D;
}
