/**
 * Minimum spanning tree over a dense distance / mutual-reachability matrix.
 *
 * HDBSCAN derives its single-linkage hierarchy from the minimum spanning tree
 * of the mutual-reachability graph. TensorFlow.js has no sparse-graph
 * primitives, so the tree is built with Prim's algorithm in plain JavaScript
 * over a dense `(n, n)` matrix — O(n²) time, O(n) auxiliary memory — which is
 * the practical scalability ceiling the rest of the density pipeline shares.
 */

/**
 * A single undirected edge of the minimum spanning tree.
 */
export interface MstEdge {
  /** One endpoint of the edge. */
  source: number;
  /** The other endpoint of the edge. */
  target: number;
  /** The edge weight (mutual-reachability / distance value). */
  weight: number;
}

/**
 * Builds the minimum spanning tree of a dense symmetric weight matrix using
 * Prim's algorithm.
 *
 * @param distance_matrix Symmetric `(n, n)` weight matrix (raw distances or
 *   mutual-reachability distances). May be a nested array or a flat row-major
 *   `Float64Array` of length `n * n`.
 * @param n Number of nodes. Required when a flat `Float64Array` is supplied;
 *   inferred from `distance_matrix.length` for nested arrays.
 * @returns The `n - 1` tree edges in the order Prim's algorithm adds them.
 *   Each edge is canonicalised so `source < target`. An empty array is
 *   returned for `n <= 1`.
 */
export function minimum_spanning_tree(
  distance_matrix: number[][] | Float64Array,
  n?: number,
): MstEdge[] {
  const is_flat = distance_matrix instanceof Float64Array;
  const size = is_flat
    ? (n ?? Math.round(Math.sqrt(distance_matrix.length)))
    : distance_matrix.length;

  if (!Number.isInteger(size) || size < 0) {
    throw new Error('Could not determine a valid node count for the matrix.');
  }

  const at = is_flat
    ? (i: number, j: number): number =>
        (distance_matrix as Float64Array)[i * size + j]
    : (i: number, j: number): number =>
        (distance_matrix as number[][])[i][j];

  if (size <= 1) {
    return [];
  }

  // Prim's algorithm over a dense matrix.
  //  - in_tree[v]        : whether v has been absorbed into the tree
  //  - best_weight[v]    : cheapest edge connecting v to the current tree
  //  - best_source[v]    : the tree-side endpoint of that cheapest edge
  const in_tree = new Uint8Array(size);
  const best_weight = new Float64Array(size);
  const best_source = new Int32Array(size);

  best_weight.fill(Number.POSITIVE_INFINITY);
  best_source.fill(-1);

  // Seed the tree with node 0.
  best_weight[0] = 0;

  const edges: MstEdge[] = [];

  for (let iter = 0; iter < size; iter++) {
    // Pick the not-yet-absorbed node with the smallest connecting edge.
    let u = -1;
    let u_weight = Number.POSITIVE_INFINITY;
    for (let v = 0; v < size; v++) {
      if (!in_tree[v] && best_weight[v] < u_weight) {
        u_weight = best_weight[v];
        u = v;
      }
    }

    // Disconnected graph guard: no reachable node remains.
    if (u === -1) {
      throw new Error(
        'Distance matrix is disconnected; cannot build a spanning tree.',
      );
    }

    in_tree[u] = 1;

    // Record the edge that connected u to the tree (skip the seed node).
    if (best_source[u] !== -1) {
      const a = best_source[u];
      const source = a < u ? a : u;
      const target = a < u ? u : a;
      edges.push({ source, target, weight: best_weight[u] });
    }

    // Relax neighbours of u.
    for (let v = 0; v < size; v++) {
      if (in_tree[v]) continue;
      const w = at(u, v);
      if (w < best_weight[v]) {
        best_weight[v] = w;
        best_source[v] = u;
      }
    }
  }

  return edges;
}
