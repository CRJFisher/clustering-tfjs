/**
 * HDBSCAN derives its single-linkage hierarchy from the minimum spanning tree
 * of the mutual-reachability graph. TensorFlow.js has no sparse-graph
 * primitives, so the tree is built with Prim's algorithm in plain JavaScript
 * over a dense `(n, n)` matrix — O(n²) time, O(n) auxiliary memory — which is
 * the practical scalability ceiling the rest of the density pipeline shares.
 */

export interface MstEdge {
  source: number;
  target: number;
  weight: number;
}

/**
 * Edges are canonicalised so `source < target`.
 * When a flat typed array is passed without an explicit `n`, the node count
 * is inferred as `round(sqrt(length))`.
 */
export function minimum_spanning_tree(
  distance_matrix: number[][] | Float32Array | Float64Array,
  n?: number,
): MstEdge[] {
  const is_flat = !(distance_matrix instanceof Array);
  const size = is_flat
    ? (n ?? Math.round(Math.sqrt(distance_matrix.length)))
    : distance_matrix.length;

  if (!Number.isInteger(size) || size < 0) {
    throw new Error('Could not determine a valid node count for the matrix.');
  }

  const at = is_flat
    ? (i: number, j: number): number =>
        (distance_matrix as Float32Array | Float64Array)[i * size + j]
    : (i: number, j: number): number =>
        (distance_matrix as number[][])[i][j];

  if (size <= 1) {
    return [];
  }

  // best_weight[v]: cheapest known edge from v to the current tree
  // best_source[v]: tree-side endpoint of that edge
  const in_tree = new Uint8Array(size);
  const best_weight = new Float64Array(size);
  const best_source = new Int32Array(size);

  best_weight.fill(Number.POSITIVE_INFINITY);
  best_source.fill(-1);

  best_weight[0] = 0;

  const edges: MstEdge[] = [];

  for (let iter = 0; iter < size; iter++) {
    let u = -1;
    let u_weight = Number.POSITIVE_INFINITY;
    for (let v = 0; v < size; v++) {
      if (!in_tree[v] && best_weight[v] < u_weight) {
        u_weight = best_weight[v];
        u = v;
      }
    }

    if (u === -1) {
      throw new Error(
        'Distance matrix is disconnected; cannot build a spanning tree.',
      );
    }

    in_tree[u] = 1;

    if (best_source[u] !== -1) {
      const a = best_source[u];
      const source = a < u ? a : u;
      const target = a < u ? u : a;
      edges.push({ source, target, weight: best_weight[u] });
    }

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
