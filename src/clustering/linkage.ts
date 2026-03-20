/**
 * Optimized hierarchical agglomerative clustering using a stored-nearest-
 * neighbor priority queue with Lance–Williams distance updates.
 *
 * Achieves O(n²) amortized complexity by maintaining per-cluster nearest-
 * neighbor pointers. Finding the global minimum merge pair is O(n) per step
 * (scan the NN distance array), and distance updates are O(n) per merge via
 * Lance–Williams. Total work is O(n²) amortized across all merges.
 *
 * The distance matrix is stored as a flat Float64Array of size n×n with
 * index-based active tracking (Uint8Array flags). This avoids the
 * O(n²)-per-merge cost of Array.splice and is cache-friendly.
 *
 * Lance–Williams recurrence for the updated distance D(t,k) when merging
 * clusters i and j into t:
 *
 *   Single   : min(D(i,k), D(j,k))
 *   Complete : max(D(i,k), D(j,k))
 *   Average  : (n_i·D(i,k) + n_j·D(j,k)) / (n_i + n_j)
 *   Ward     : sqrt(max(((n_i+n_k)·D(i,k)² + (n_j+n_k)·D(j,k)² - n_k·D(i,j)²)
 *                       / (n_i + n_j + n_k), 0))
 */

export type LinkageCriterion = 'single' | 'complete' | 'average' | 'ward';

/**
 * Record of a single merge operation in the agglomeration process.
 */
export interface MergeRecord {
  /** Index of the surviving cluster (min of the two merged indices). */
  clusterA: number;
  /** Index of the removed cluster (max of the two merged indices). */
  clusterB: number;
  /** Distance at which the merge occurred. */
  distance: number;
  /** Size of the merged cluster after the merge. */
  newSize: number;
}

/**
 * Computes the Lance–Williams updated distance between the newly merged
 * cluster t = (i ∪ j) and another cluster k.
 */
function lancewilliams(
  dik: number,
  djk: number,
  dij: number,
  ni: number,
  nj: number,
  nk: number,
  linkage: LinkageCriterion,
): number {
  switch (linkage) {
    case 'single':
      return Math.min(dik, djk);
    case 'complete':
      return Math.max(dik, djk);
    case 'average':
      return (ni * dik + nj * djk) / (ni + nj);
    case 'ward': {
      const total = ni + nj + nk;
      const numerator =
        (ni + nk) * dik * dik +
        (nj + nk) * djk * djk -
        nk * dij * dij;
      return Math.sqrt(Math.max(numerator / total, 0));
    }
  }
}

/**
 * Scans all active clusters to find the nearest neighbor of cluster `idx`.
 * Returns the NN index and distance.
 */
function findNearestNeighbor(
  D: Float64Array,
  n: number,
  active: Uint8Array,
  idx: number,
): { nn: number; dist: number } {
  let bestDist = Infinity;
  let bestNN = -1;
  const row = idx * n;
  for (let k = 0; k < n; k++) {
    if (!active[k] || k === idx) continue;
    const d = D[row + k];
    if (d < bestDist) {
      bestDist = d;
      bestNN = k;
    }
  }
  return { nn: bestNN, dist: bestDist };
}

/**
 * Runs agglomerative clustering using a stored-nearest-neighbor priority
 * queue with Lance–Williams distance updates.
 *
 * For each active cluster, the nearest neighbor and its distance are cached.
 * The global minimum merge pair is found by scanning these cached distances
 * in O(n) time. After each merge, the cache is updated in O(n) amortized
 * time, yielding O(n²) total complexity.
 *
 * @param D     Flat n×n distance matrix (Float64Array). Mutated in place.
 * @param n     Number of original data points.
 * @param nClusters Target number of clusters.
 * @param linkage   Linkage criterion.
 * @returns Array of MergeRecord describing each merge in order.
 */
export function heapCluster(
  D: Float64Array,
  n: number,
  nClusters: number,
  linkage: LinkageCriterion,
): MergeRecord[] {
  const active = new Uint8Array(n);
  active.fill(1);

  const clusterSizes = new Float64Array(n);
  clusterSizes.fill(1);

  // Per-cluster nearest neighbor cache (the "priority queue")
  const nn = new Int32Array(n);      // nn[i] = nearest neighbor of i
  const nnDist = new Float64Array(n); // nnDist[i] = distance to nn[i]

  // Initialize NN cache — O(n²)
  for (let i = 0; i < n; i++) {
    const result = findNearestNeighbor(D, n, active, i);
    nn[i] = result.nn;
    nnDist[i] = result.dist;
  }

  const merges: MergeRecord[] = [];
  const targetMerges = n - nClusters;

  while (merges.length < targetMerges) {
    // Find the active cluster with the smallest NN distance — O(n)
    let minDist = Infinity;
    let minI = -1;
    for (let i = 0; i < n; i++) {
      if (!active[i]) continue;
      if (nnDist[i] < minDist) {
        minDist = nnDist[i];
        minI = i;
      }
    }

    const j = nn[minI];
    const survivor = Math.min(minI, j);
    const removed = Math.max(minI, j);
    const mergeDist = D[survivor * n + removed];

    merges.push({
      clusterA: survivor,
      clusterB: removed,
      distance: mergeDist,
      newSize: clusterSizes[survivor] + clusterSizes[removed],
    });

    // Update distances from survivor to all other active clusters
    const dij = mergeDist;
    const ni = clusterSizes[survivor];
    const nj = clusterSizes[removed];

    // Deactivate the removed cluster first
    active[removed] = 0;

    for (let k = 0; k < n; k++) {
      if (!active[k] || k === survivor) continue;

      const newDist = lancewilliams(
        D[survivor * n + k],
        D[removed * n + k],
        dij,
        ni,
        nj,
        clusterSizes[k],
        linkage,
      );
      D[survivor * n + k] = newDist;
      D[k * n + survivor] = newDist;

      // Update k's NN cache
      if (nn[k] === removed || nn[k] === survivor) {
        // k's NN was involved in the merge — need full rescan
        const result = findNearestNeighbor(D, n, active, k);
        nn[k] = result.nn;
        nnDist[k] = result.dist;
      } else if (newDist < nnDist[k]) {
        // The merged cluster is now closer to k than k's current NN
        nn[k] = survivor;
        nnDist[k] = newDist;
      }
    }

    clusterSizes[survivor] += clusterSizes[removed];

    // Rescan survivor's NN
    const result = findNearestNeighbor(D, n, active, survivor);
    nn[survivor] = result.nn;
    nnDist[survivor] = result.dist;
  }

  return merges;
}
