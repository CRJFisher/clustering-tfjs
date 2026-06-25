/**
 * Optimized hierarchical agglomerative clustering using the nearest-neighbor
 * chain algorithm with Lance–Williams distance updates.
 *
 * NN-chain follows reciprocal nearest neighbors until it finds a reducible
 * merge pair, then updates distances in place. For single, complete, average,
 * and Ward linkage this gives guaranteed O(n²) time and O(n²) memory.
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
  /** Lower active slot index of the two merged clusters. */
  cluster_a: number;
  /** Higher active slot index of the two merged clusters. */
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
      // The factor t = 1/(ni+nj+nk) is distributed into each term rather than
      // dividing a summed numerator once. The two forms are algebraically
      // identical but round differently in IEEE-754: summing first and dividing
      // last can shift the result by one ULP, which flips exact distance ties
      // on symmetric data and produces a different (still valid) dendrogram.
      // This arrangement is bit-identical to scipy's `_ward`, so NN-chain
      // reproduces scipy/sklearn ward trees exactly, ties included.
      const t = 1 / (ni + nj + nk);
      return Math.sqrt(
        Math.max(
          (ni + nk) * t * dik * dik +
            (nj + nk) * t * djk * djk -
            nk * t * dij * dij,
          0,
        ),
      );
    }
  }
}

/**
 * Runs agglomerative clustering using the nearest-neighbor chain algorithm
 * with Lance–Williams distance updates.
 *
 * The emitted merges are sorted by distance before returning, matching the
 * scipy/fastcluster convention for NN-chain. The raw discovery order is not a
 * valid dendrogram order for cutting.
 *
 * @param D Flat n×n distance matrix (Float64Array). Mutated in place.
 * @param n Number of original data points.
 * @param linkage   Linkage criterion.
 * @returns Array of MergeRecord describing the full tree in distance order.
 */
export function nn_chain_cluster(
  D: Float64Array,
  n: number,
  linkage: LinkageCriterion,
): MergeRecord[] {
  const active = new Uint8Array(n);
  active.fill(1);

  const cluster_sizes = new Float64Array(n);
  cluster_sizes.fill(1);

  const chain = new Int32Array(n);
  let chain_len = 0;
  const merges: MergeRecord[] = [];

  while (merges.length < n - 1) {
    if (chain_len === 0) {
      for (let i = 0; i < n; i++) {
        if (active[i]) {
          chain[0] = i;
          chain_len = 1;
          break;
        }
      }
    }

    let x = -1;
    let y = -1;
    let min_dist = Infinity;

    while (true) {
      x = chain[chain_len - 1];

      // Prefer the previous chain element on ties to avoid cycles, matching
      // scipy's strict-inequality NN-chain rule.
      if (chain_len > 1) {
        y = chain[chain_len - 2];
        min_dist = D[x * n + y];
      } else {
        y = -1;
        min_dist = Infinity;
      }

      const row = x * n;
      for (let i = 0; i < n; i++) {
        if (!active[i] || i === x) continue;
        const dist = D[row + i];
        if (dist < min_dist) {
          min_dist = dist;
          y = i;
        }
      }

      if (chain_len > 1 && y === chain[chain_len - 2]) {
        break;
      }

      chain[chain_len] = y;
      chain_len++;
    }

    chain_len -= 2;

    const lower = Math.min(x, y);
    const higher = Math.max(x, y);
    const removed = lower;
    const survivor = higher;

    const ni = cluster_sizes[removed];
    const nj = cluster_sizes[survivor];
    const new_size = ni + nj;

    active[removed] = 0;
    cluster_sizes[removed] = 0;

    for (let k = 0; k < n; k++) {
      if (!active[k] || k === survivor) continue;

      const new_dist = lance_williams(
        D[removed * n + k],
        D[survivor * n + k],
        min_dist,
        ni,
        nj,
        cluster_sizes[k],
        linkage,
      );
      D[survivor * n + k] = new_dist;
      D[k * n + survivor] = new_dist;
    }

    cluster_sizes[survivor] = new_size;

    merges.push({
      cluster_a: lower,
      cluster_b: higher,
      distance: min_dist,
      new_size,
    });
  }

  return merges.sort((a, b) => a.distance - b.distance);
}
