/**
 * Per-point core (k-distance) computation for density-based clustering.
 *
 * The k-distance of a point is the distance to its k-th nearest neighbour,
 * where the point itself counts as its own first nearest neighbour (distance
 * 0). This matches scikit-learn's HDBSCAN core-distance definition, which
 * takes `core_distance(i) = neighbors_distances[i, min_samples - 1]` from a
 * `kneighbors` query of size `min_samples` whose first column is the point
 * itself.
 */

/**
 * Computes the per-point k-distance vector from a k-nearest-neighbour scan.
 *
 * @param neighbor_distances Per-point neighbour distances in nearest-first
 *   order, where index 0 of each row is the point itself (distance 0). This is
 *   the `neighbor_distances` field produced by the k-NN scan in
 *   `compute_knn_affinity`, or the sorted rows of a full distance matrix.
 * @param k The neighbourhood size (`min_samples`). The point itself counts as
 *   the first neighbour, so the returned value is the distance to the
 *   `(k - 1)`-th *other* nearest neighbour.
 * @returns A `Float64Array` of length `n` holding each point's k-distance.
 * @throws If `k` is not a positive integer, or any row has fewer than `k`
 *   neighbours (the scan retained too few neighbours for the requested `k`).
 */
export function kdistance(
  neighbor_distances: number[][],
  k: number,
): Float64Array {
  if (!Number.isInteger(k) || k < 1) {
    throw new Error('k must be a positive integer (>= 1).');
  }

  const n = neighbor_distances.length;
  const core = new Float64Array(n);

  for (let i = 0; i < n; i++) {
    const row = neighbor_distances[i];
    if (row.length < k) {
      throw new Error(
        `Point ${i} has only ${row.length} neighbours but k=${k} requested. ` +
          'Increase the neighbourhood size of the k-NN scan.',
      );
    }
    core[i] = row[k - 1];
  }

  return core;
}
