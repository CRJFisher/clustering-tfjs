/**
 * Optimized hierarchical agglomerative clustering using a stored-nearest-
 * neighbor priority queue with Lance–Williams distance updates.
 *
 * Maintains per-cluster nearest-neighbor pointers (the stored-NN / Anderberg
 * scheme). Finding the global minimum merge pair is O(n) per step (scan the NN
 * distance array), and distance updates are O(n) per merge via Lance–Williams.
 * Typical (average-case) complexity is O(n²); the worst case is O(n³) when a
 * large fraction of clusters must re-scan their nearest neighbor after each
 * merge.
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
  cluster_a: number;
  /** Index of the removed cluster (max of the two merged indices). */
  cluster_b: number;
  /** Distance at which the merge occurred. */
  distance: number;
  /** Size of the merged cluster after the merge. */
  new_size: number;
}

/**
 * Computes the Lance–Williams updated distance between the newly merged
 * cluster t = (i ∪ j) and another cluster k.
 */
function lance_williams(
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
 * Scans all active clusters to find the nearest neighbor of cluster `idx`
 * and writes the result directly into the nn/nn_dist arrays (zero allocation).
 */
function update_nearest_neighbor(
  D: Float64Array,
  n: number,
  active: Uint8Array,
  idx: number,
  nn: Int32Array,
  nn_dist: Float64Array,
): void {
  let best_dist = Infinity;
  let best_nn = -1;
  const row = idx * n;
  for (let k = 0; k < n; k++) {
    if (!active[k] || k === idx) continue;
    const d = D[row + k];
    if (d < best_dist) {
      best_dist = d;
      best_nn = k;
    }
  }
  nn[idx] = best_nn;
  nn_dist[idx] = best_dist;
}

/**
 * Runs agglomerative clustering using a stored-nearest-neighbor priority
 * queue with Lance–Williams distance updates.
 *
 * For each active cluster, the nearest neighbor and its distance are cached.
 * The global minimum merge pair is found by scanning these cached distances
 * in O(n) time. After each merge, the cache is updated incrementally, giving
 * O(n²) total complexity in the typical case (O(n³) worst case when most
 * clusters must re-scan their nearest neighbor every merge).
 *
 * @param D     Flat n×n distance matrix (Float64Array). Mutated in place.
 * @param n     Number of original data points.
 * @param n_clusters Target number of clusters.
 * @param linkage   Linkage criterion.
 * @returns Array of MergeRecord describing each merge in order.
 */
export function stored_nn_cluster(
  D: Float64Array,
  n: number,
  n_clusters: number,
  linkage: LinkageCriterion,
): MergeRecord[] {
  const active = new Uint8Array(n);
  active.fill(1);

  const cluster_sizes = new Float64Array(n);
  cluster_sizes.fill(1);

  // Per-cluster nearest neighbor cache (the "priority queue")
  const nn = new Int32Array(n);      // nn[i] = nearest neighbor of i
  const nn_dist = new Float64Array(n); // nn_dist[i] = distance to nn[i]

  // Initialize NN cache — O(n²)
  for (let i = 0; i < n; i++) {
    update_nearest_neighbor(D, n, active, i, nn, nn_dist);
  }

  const merges: MergeRecord[] = [];
  const target_merges = n - n_clusters;

  while (merges.length < target_merges) {
    // Find the active cluster with the smallest NN distance — O(n)
    let min_dist = Infinity;
    let min_i = -1;
    for (let i = 0; i < n; i++) {
      if (!active[i]) continue;
      if (nn_dist[i] < min_dist) {
        min_dist = nn_dist[i];
        min_i = i;
      }
    }

    if (min_i === -1) break; // no active clusters remain

    const j = nn[min_i];
    const survivor = Math.min(min_i, j);
    const removed = Math.max(min_i, j);
    const merge_dist = D[survivor * n + removed];

    merges.push({
      cluster_a: survivor,
      cluster_b: removed,
      distance: merge_dist,
      new_size: cluster_sizes[survivor] + cluster_sizes[removed],
    });

    // Update distances from survivor to all other active clusters
    const dij = merge_dist;
    const ni = cluster_sizes[survivor];
    const nj = cluster_sizes[removed];

    // Deactivate the removed cluster first
    active[removed] = 0;

    for (let k = 0; k < n; k++) {
      if (!active[k] || k === survivor) continue;

      const new_dist = lance_williams(
        D[survivor * n + k],
        D[removed * n + k],
        dij,
        ni,
        nj,
        cluster_sizes[k],
        linkage,
      );
      D[survivor * n + k] = new_dist;
      D[k * n + survivor] = new_dist;

      // Update k's NN cache
      if (nn[k] === removed || nn[k] === survivor) {
        // k's NN was involved in the merge — need full rescan
        update_nearest_neighbor(D, n, active, k, nn, nn_dist);
      } else if (new_dist < nn_dist[k]) {
        // The merged cluster is now closer to k than k's current NN
        nn[k] = survivor;
        nn_dist[k] = new_dist;
      }
    }

    cluster_sizes[survivor] += cluster_sizes[removed];

    // Rescan survivor's NN
    update_nearest_neighbor(D, n, active, survivor, nn, nn_dist);
  }

  return merges;
}
