/**
 * Matches scikit-learn's HDBSCAN `core_distance(i) = neighbors_distances[i, min_samples - 1]`
 * convention: the point itself occupies index 0 (distance 0).
 *
 * @param neighbor_distances Per-point neighbour distances in nearest-first
 *   order, where index 0 of each row is the point itself (distance 0). This is
 *   the `neighbor_distances` field produced by the k-NN scan in
 *   `compute_knn_affinity`, or the sorted rows of a full distance matrix.
 * @param k The neighbourhood size (`min_samples`). The point itself counts as
 *   the first neighbour, so the returned value is the distance to the
 *   `(k - 1)`-th *other* nearest neighbour.
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
