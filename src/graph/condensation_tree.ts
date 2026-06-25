import type { MstEdge } from './minimum_spanning_tree';

/**
 * λ (lambda) is the inverse of the merge distance: dense structure persists to
 * high λ. This module operates purely on graph structures and is independent of
 * any estimator wrapper.
 */

export interface CondensedEdge {
  parent: number;
  /** Child cluster id (>= n_samples) or point index (< n_samples). */
  child: number;
  /** λ at which the child leaves the parent (1 / merge distance). */
  lambda_val: number;
  /** Population of the child (1 for points, cluster size for clusters). */
  child_size: number;
}

export interface ClusterSelectionOptions {
  cluster_selection_method?: 'eom' | 'leaf';
  cluster_selection_epsilon?: number;
  allow_single_cluster?: boolean;
}

export interface CondensedClustering {
  labels: number[];
  probabilities: number[];
  /** Most-persistent point index per cluster label. */
  exemplar_indices: Map<number, number>;
}

/**
 * @returns `n_samples - 1` rows `[left, right, distance, size]`, where `left`
 *   and `right` are node ids (points `< n_samples`, merged clusters `>=
 *   n_samples`) and `size` is the merged population.
 */
export function build_single_linkage(
  mst_edges: MstEdge[],
  n_samples: number,
): number[][] {
  const edges = [...mst_edges].sort((a, b) => a.weight - b.weight);

  const total_nodes = 2 * n_samples - 1;
  const parent = new Int32Array(Math.max(total_nodes, 1)).fill(-1);
  const size = new Int32Array(Math.max(total_nodes, 1));
  for (let i = 0; i < n_samples; i++) size[i] = 1;

  const find = (x: number): number => {
    let root = x;
    while (parent[root] !== -1) root = parent[root];
    let node = x;
    while (node !== root) {
      const next = parent[node];
      parent[node] = root;
      node = next;
    }
    return root;
  };

  let next_label = n_samples;
  const result: number[][] = [];
  for (const e of edges) {
    const a = find(e.source);
    const b = find(e.target);
    const merged = size[a] + size[b];
    result.push([a, b, e.weight, merged]);
    parent[a] = next_label;
    parent[b] = next_label;
    size[next_label] = merged;
    next_label++;
  }
  return result;
}

function bfs_hierarchy(
  hierarchy: number[][],
  bfs_root: number,
  n_samples: number,
): number[] {
  let to_process = [bfs_root];
  const result: number[] = [];
  while (to_process.length > 0) {
    result.push(...to_process);
    const next: number[] = [];
    for (const node of to_process) {
      if (node >= n_samples) {
        const children = hierarchy[node - n_samples];
        next.push(children[0], children[1]);
      }
    }
    to_process = next;
  }
  return result;
}

/**
 * Components smaller than `min_cluster_size` are treated as points falling out
 * of their parent cluster rather than as distinct clusters; a split into two
 * sufficiently large children creates two new clusters, and a one-sided split
 * lets the surviving side continue under the same cluster id.
 */
export function build_condensation_tree(
  mst_edges: MstEdge[],
  n_samples: number,
  min_cluster_size: number,
): CondensedEdge[] {
  const hierarchy = build_single_linkage(mst_edges, n_samples);
  return condense_hierarchy(hierarchy, n_samples, min_cluster_size);
}

export function condense_hierarchy(
  hierarchy: number[][],
  n_samples: number,
  min_cluster_size: number,
): CondensedEdge[] {
  const mcs = Math.max(2, Math.floor(min_cluster_size));

  if (hierarchy.length === 0) {
    return [];
  }

  const root = 2 * hierarchy.length; // 2*(n-1) = 2n-2
  let next_label = n_samples + 1;

  const node_list = bfs_hierarchy(hierarchy, root, n_samples);
  const relabel = new Int32Array(root + 1);
  relabel[root] = n_samples;
  const ignore = new Uint8Array(root + 1);
  const result: CondensedEdge[] = [];

  for (const node of node_list) {
    if (ignore[node] || node < n_samples) continue;

    const children = hierarchy[node - n_samples];
    const left = children[0];
    const right = children[1];
    const dist = children[2];
    const lambda_value = dist > 0 ? 1 / dist : Number.POSITIVE_INFINITY;

    const left_count =
      left >= n_samples ? hierarchy[left - n_samples][3] : 1;
    const right_count =
      right >= n_samples ? hierarchy[right - n_samples][3] : 1;

    if (left_count >= mcs && right_count >= mcs) {
      relabel[left] = next_label++;
      result.push({
        parent: relabel[node],
        child: relabel[left],
        lambda_val: lambda_value,
        child_size: left_count,
      });
      relabel[right] = next_label++;
      result.push({
        parent: relabel[node],
        child: relabel[right],
        lambda_val: lambda_value,
        child_size: right_count,
      });
    } else if (left_count < mcs && right_count < mcs) {
      for (const sub of bfs_hierarchy(hierarchy, left, n_samples)) {
        if (sub < n_samples) {
          result.push({
            parent: relabel[node],
            child: sub,
            lambda_val: lambda_value,
            child_size: 1,
          });
        }
        ignore[sub] = 1;
      }
      for (const sub of bfs_hierarchy(hierarchy, right, n_samples)) {
        if (sub < n_samples) {
          result.push({
            parent: relabel[node],
            child: sub,
            lambda_val: lambda_value,
            child_size: 1,
          });
        }
        ignore[sub] = 1;
      }
    } else if (left_count < mcs) {
      relabel[right] = relabel[node];
      for (const sub of bfs_hierarchy(hierarchy, left, n_samples)) {
        if (sub < n_samples) {
          result.push({
            parent: relabel[node],
            child: sub,
            lambda_val: lambda_value,
            child_size: 1,
          });
        }
        ignore[sub] = 1;
      }
    } else {
      relabel[left] = relabel[node];
      for (const sub of bfs_hierarchy(hierarchy, right, n_samples)) {
        if (sub < n_samples) {
          result.push({
            parent: relabel[node],
            child: sub,
            lambda_val: lambda_value,
            child_size: 1,
          });
        }
        ignore[sub] = 1;
      }
    }
  }

  return result;
}

/** Births: λ at which each cluster is born (the row where it is a child). */
function compute_births(
  tree: CondensedEdge[],
  n_samples: number,
): Map<number, number> {
  const births = new Map<number, number>();
  for (const e of tree) births.set(e.child, e.lambda_val);
  births.set(n_samples, 0);
  return births;
}

/**
 * Computes per-cluster stability: `Σ (λ_child - λ_birth(cluster)) * child_size`
 * over all rows whose parent is the cluster.
 */
export function compute_stability(
  tree: CondensedEdge[],
  n_samples: number,
): Map<number, number> {
  const births = compute_births(tree, n_samples);
  const stability = new Map<number, number>();
  for (const e of tree) {
    const birth = births.get(e.parent) ?? 0;
    const prev = stability.get(e.parent) ?? 0;
    stability.set(e.parent, prev + (e.lambda_val - birth) * e.child_size);
  }
  return stability;
}

function cluster_children(tree: CondensedEdge[]): Map<number, number[]> {
  const children = new Map<number, number[]>();
  for (const e of tree) {
    if (e.child_size > 1) {
      if (!children.has(e.parent)) children.set(e.parent, []);
      children.get(e.parent)!.push(e.child);
    }
  }
  return children;
}

function bfs_clusters(
  children_of: Map<number, number[]>,
  node: number,
): number[] {
  const result: number[] = [];
  let queue = [node];
  while (queue.length > 0) {
    result.push(...queue);
    const next: number[] = [];
    for (const c of queue) {
      const ch = children_of.get(c);
      if (ch) next.push(...ch);
    }
    queue = next;
  }
  return result;
}

function traverse_upwards(
  births: Map<number, number>,
  cluster_parent: Map<number, number>,
  epsilon: number,
  leaf: number,
  root: number,
  allow_single_cluster: boolean,
): number {
  const parent = cluster_parent.get(leaf);
  if (parent === undefined || parent === root) {
    return allow_single_cluster && parent === root ? root : leaf;
  }
  const parent_lambda = births.get(parent) ?? Number.POSITIVE_INFINITY;
  const parent_eps = parent_lambda > 0 ? 1 / parent_lambda : Number.POSITIVE_INFINITY;
  // sklearn stops at the first ancestor born at distance >= epsilon.
  if (parent_eps >= epsilon) {
    return parent;
  }
  return traverse_upwards(
    births,
    cluster_parent,
    epsilon,
    parent,
    root,
    allow_single_cluster,
  );
}

function epsilon_search(
  leaves: Set<number>,
  births: Map<number, number>,
  cluster_parent: Map<number, number>,
  children_of: Map<number, number[]>,
  epsilon: number,
  root: number,
  allow_single_cluster: boolean,
): Set<number> {
  const selected: number[] = [];
  const processed = new Set<number>();

  for (const leaf of leaves) {
    const birth_lambda = births.get(leaf) ?? Number.POSITIVE_INFINITY;
    const eps = birth_lambda > 0 ? 1 / birth_lambda : Number.POSITIVE_INFINITY;
    if (eps < epsilon) {
      if (!processed.has(leaf)) {
        const epsilon_child = traverse_upwards(
          births,
          cluster_parent,
          epsilon,
          leaf,
          root,
          allow_single_cluster,
        );
        selected.push(epsilon_child);
        for (const sub of bfs_clusters(children_of, epsilon_child)) {
          processed.add(sub);
        }
      }
    } else {
      selected.push(leaf);
    }
  }

  return new Set(selected);
}

/**
 * Selects the flat set of clusters from the condensed tree.
 *
 * `'eom'` (Excess of Mass) keeps a cluster when its own stability exceeds the
 * summed stability of its selected descendants; `'leaf'` keeps every leaf
 * cluster. `cluster_selection_epsilon` then merges clusters whose birth distance
 * (`1 / birth_lambda`) is below `epsilon` into a coarser ancestor.
 *
 * @returns The set of selected cluster ids.
 */
export function excess_of_mass(
  tree: CondensedEdge[],
  n_samples: number,
  options: ClusterSelectionOptions = {},
): Set<number> {
  const {
    cluster_selection_method = 'eom',
    cluster_selection_epsilon = 0,
    allow_single_cluster = false,
  } = options;

  if (tree.length === 0) return new Set();

  const root = n_samples;
  const births = compute_births(tree, n_samples);
  const children_of = cluster_children(tree);
  const cluster_parent = new Map<number, number>();
  for (const e of tree) {
    if (e.child_size > 1) cluster_parent.set(e.child, e.parent);
  }

  const all_clusters = new Set<number>();
  for (const e of tree) if (e.child_size > 1) all_clusters.add(e.child);
  if (allow_single_cluster) all_clusters.add(root);

  let selected: Set<number>;

  if (cluster_selection_method === 'leaf') {
    const parents = new Set<number>();
    for (const c of all_clusters) {
      if (children_of.has(c) && (children_of.get(c)?.length ?? 0) > 0) {
        parents.add(c);
      }
    }
    let leaves = new Set<number>(
      [...all_clusters].filter((c) => !parents.has(c)),
    );
    if (leaves.size === 0) leaves = new Set([root]);
    selected =
      cluster_selection_epsilon > 0
        ? epsilon_search(
            leaves,
            births,
            cluster_parent,
            children_of,
            cluster_selection_epsilon,
            root,
            allow_single_cluster,
          )
        : leaves;
  } else {
    const stability = compute_stability(tree, n_samples);
    const stab = new Map(stability);
    let nodes = [...stability.keys()].sort((a, b) => b - a);
    if (!allow_single_cluster) nodes = nodes.filter((c) => c !== root);

    const is_cluster = new Map<number, boolean>();
    for (const c of nodes) is_cluster.set(c, true);

    for (const node of nodes) {
      const ch = children_of.get(node) ?? [];
      let subtree = 0;
      for (const c of ch) subtree += stab.get(c) ?? 0;
      if (subtree > (stab.get(node) ?? 0)) {
        is_cluster.set(node, false);
        stab.set(node, subtree);
      } else {
        for (const sub of bfs_clusters(children_of, node)) {
          if (sub !== node) is_cluster.set(sub, false);
        }
      }
    }

    const eom = new Set<number>(
      [...is_cluster].filter(([, v]) => v).map(([k]) => k),
    );
    selected =
      cluster_selection_epsilon > 0
        ? epsilon_search(
            eom,
            births,
            cluster_parent,
            children_of,
            cluster_selection_epsilon,
            root,
            allow_single_cluster,
          )
        : eom;
  }

  return selected;
}

/**
 * Points are routed to the lowest selected ancestor of the cluster they fall
 * out of; points with no selected ancestor are noise (`-1`).
 */
export function extract_labels(
  tree: CondensedEdge[],
  selected: Set<number>,
  n_samples: number,
  allow_single_cluster = false,
): CondensedClustering {
  const root = n_samples;

  const point_parent = new Int32Array(n_samples).fill(-1);
  const point_lambda = new Float64Array(n_samples);
  const cluster_parent = new Map<number, number>();
  const deaths = new Map<number, number>();

  for (const e of tree) {
    deaths.set(e.parent, Math.max(deaths.get(e.parent) ?? 0, e.lambda_val));
    if (e.child < n_samples) {
      point_parent[e.child] = e.parent;
      point_lambda[e.child] = e.lambda_val;
    } else if (e.child_size > 1) {
      cluster_parent.set(e.child, e.parent);
    }
  }

  const clusters_sorted = [...selected].sort((a, b) => a - b);
  const label_map = new Map<number, number>();
  clusters_sorted.forEach((c, i) => label_map.set(c, i));

  const labels = new Array<number>(n_samples).fill(-1);

  for (let p = 0; p < n_samples; p++) {
    let c = point_parent[p];
    if (c === -1) continue;
    while (!selected.has(c) && c !== root) {
      const next = cluster_parent.get(c);
      if (next === undefined) break;
      c = next;
    }
    if (selected.has(c)) {
      // Root is only assignable when single-cluster output is allowed.
      if (c !== root || allow_single_cluster) {
        labels[p] = label_map.get(c)!;
      }
    }
  }

  const probabilities = new Array<number>(n_samples).fill(0);
  const exemplar_indices = new Map<number, number>();
  const best_lambda = new Map<number, number>();

  for (let p = 0; p < n_samples; p++) {
    const lab = labels[p];
    if (lab === -1) continue;

    const cluster = clusters_sorted[lab];
    const max_lambda = deaths.get(cluster) ?? 0;
    const lam = point_lambda[p];

    if (max_lambda === 0 || !Number.isFinite(lam)) {
      probabilities[p] = 1.0;
    } else {
      probabilities[p] = Math.min(lam, max_lambda) / max_lambda;
    }

    if (!best_lambda.has(lab) || lam > best_lambda.get(lab)!) {
      best_lambda.set(lab, lam);
      exemplar_indices.set(lab, p);
    }
  }

  return { labels, probabilities, exemplar_indices };
}
