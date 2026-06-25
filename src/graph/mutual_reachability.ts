/**
 * Mutual-reachability distance for density-based clustering.
 *
 * The mutual-reachability distance between two points smooths the raw distance
 * by each point's core (k-)distance:
 *
 *   d_mreach(i, j) = max(core_distance_i, core_distance_j, distance_i_j)
 *
 * Connecting points by mutual reachability rather than raw distance is what
 * makes HDBSCAN robust to single-linkage chaining through sparse regions: a
 * point in a sparse region has a large core distance, so edges through it are
 * inflated and it is harder for it to bridge dense clusters.
 */

/**
 * Builds the dense mutual-reachability matrix from a pairwise distance matrix
 * and per-point core distances.
 *
 * @param distance_matrix Symmetric `(n, n)` pairwise distance matrix.
 * @param core_distances Per-point core distances of length `n` (see
 *   {@link kdistance}).
 * @returns An `(n, n)` matrix `M` where `M[i][j] = max(core_i, core_j, d_ij)`.
 *   The diagonal is `max(core_i, 0) = core_i`.
 * @throws If the dimensions of `distance_matrix` and `core_distances` disagree.
 */
export function mutual_reachability(
  distance_matrix: number[][],
  core_distances: ArrayLike<number>,
): number[][] {
  const n = distance_matrix.length;
  if (core_distances.length !== n) {
    throw new Error(
      `core_distances length (${core_distances.length}) must match the ` +
        `distance matrix size (${n}).`,
    );
  }

  const result: number[][] = new Array<number[]>(n);
  for (let i = 0; i < n; i++) {
    const row = distance_matrix[i];
    if (row.length !== n) {
      throw new Error(
        `distance_matrix must be square; row ${i} has length ${row.length}.`,
      );
    }
    const core_i = core_distances[i];
    const out = new Array<number>(n);
    for (let j = 0; j < n; j++) {
      const core_j = core_distances[j];
      const d = row[j];
      let m = core_i > core_j ? core_i : core_j;
      if (d > m) m = d;
      out[j] = m;
    }
    result[i] = out;
  }

  return result;
}
