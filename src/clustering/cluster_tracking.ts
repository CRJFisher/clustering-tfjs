import * as tf from '../backend/adapter';
import { pairwise_distance_matrix } from '../distance/pairwise_distance';

/**
 * Cross-snapshot cluster tracking.
 *
 * Clustering a drifting/streaming dataset produces a fresh set of clusters at
 * each snapshot with no identity across time. `track_clusters` compares two
 * consecutive snapshots by their representative vectors, computes an optimal
 * bipartite assignment, and classifies each cluster's fate as a transition
 * (`PERSIST`, `EMERGE`, `DIE`, `MERGE`, `SPLIT`) with a stable lifeline id
 * carried forward across snapshots.
 *
 * The function is **stateless**: the caller owns the {@link TrackingState} and
 * threads it between frames, so tracking composes with any clustering pipeline
 * without hidden global state.
 */

export type TransitionType = 'PERSIST' | 'EMERGE' | 'DIE' | 'MERGE' | 'SPLIT';

/** A single cross-snapshot transition event. */
export interface ClusterTransition {
  type: TransitionType;
  /** Previous-snapshot cluster indices involved. */
  prev: number[];
  /** Current-snapshot cluster indices involved. */
  curr: number[];
  /** Stable lifeline id associated with the event. */
  lifeline_id: number;
}

/** Lifeline state for a snapshot, threaded between frames by the caller. */
export interface TrackingState {
  /** Lifeline id per cluster in this snapshot. */
  lifelines: number[];
  /** Next unused lifeline id. */
  next_lifeline_id: number;
}

export interface ClusterTrackingOptions {
  /**
   * Similarity threshold in `[0, 1]`. A previous/current pair is a candidate
   * match only when cosine similarity `>= threshold` (cosine cost
   * `<= 1 - threshold`). Default 0.5.
   */
  threshold?: number;
}

export interface TrackingResult {
  /** Optimal assignment: `assignment[j]` = matched prev index for curr `j`, or -1. */
  assignment: number[];
  /** Cosine cost matrix, shape `n_prev × n_curr`. */
  cost_matrix: number[][];
  /** Emitted transition events. */
  transitions: ClusterTransition[];
  /** Lifeline state for the current snapshot; thread this into the next call. */
  state: TrackingState;
}

/**
 * Solves the minimum-cost assignment on a square matrix (Hungarian / Jonker–
 * Volgenant potentials, O(n³)). Returns `row -> col` assignment.
 */
function hungarian_square(cost: number[][]): number[] {
  const n = cost.length;
  const INF = Number.POSITIVE_INFINITY;
  // 1-indexed potentials (e-maxx formulation).
  const u = new Array<number>(n + 1).fill(0);
  const v = new Array<number>(n + 1).fill(0);
  const p = new Array<number>(n + 1).fill(0);
  const way = new Array<number>(n + 1).fill(0);

  for (let i = 1; i <= n; i++) {
    p[0] = i;
    let j0 = 0;
    const minv = new Array<number>(n + 1).fill(INF);
    const used = new Array<boolean>(n + 1).fill(false);
    do {
      used[j0] = true;
      const i0 = p[j0];
      let delta = INF;
      let j1 = -1;
      for (let j = 1; j <= n; j++) {
        if (used[j]) continue;
        const cur = cost[i0 - 1][j - 1] - u[i0] - v[j];
        if (cur < minv[j]) {
          minv[j] = cur;
          way[j] = j0;
        }
        if (minv[j] < delta) {
          delta = minv[j];
          j1 = j;
        }
      }
      for (let j = 0; j <= n; j++) {
        if (used[j]) {
          u[p[j]] += delta;
          v[j] -= delta;
        } else {
          minv[j] -= delta;
        }
      }
      j0 = j1;
    } while (p[j0] !== 0);
    do {
      const j1 = way[j0];
      p[j0] = p[j1];
      j0 = j1;
    } while (j0);
  }

  const row_to_col = new Array<number>(n).fill(-1);
  for (let j = 1; j <= n; j++) {
    if (p[j] > 0) row_to_col[p[j] - 1] = j - 1;
  }
  return row_to_col;
}

/**
 * Optimal min-cost assignment of a rectangular cost matrix, matching
 * `min(n_prev, n_curr)` pairs (scipy `linear_sum_assignment` semantics).
 * Returns `prev -> curr` index per prev row (`-1` if unmatched).
 */
function linear_sum_assignment(cost: number[][]): number[] {
  const n_prev = cost.length;
  const n_curr = n_prev > 0 ? cost[0].length : 0;
  if (n_prev === 0 || n_curr === 0) {
    return new Array<number>(n_prev).fill(-1);
  }

  const dim = Math.max(n_prev, n_curr);
  let max_cost = 0;
  for (let i = 0; i < n_prev; i++) {
    for (let j = 0; j < n_curr; j++) {
      if (cost[i][j] > max_cost) max_cost = cost[i][j];
    }
  }
  // Pad to square with a constant larger than any real cost so the optimal
  // square assignment selects the real min-cost pairs first.
  const pad = max_cost + 1;
  const square: number[][] = Array.from({ length: dim }, (_v, i) =>
    Array.from({ length: dim }, (_w, j) =>
      i < n_prev && j < n_curr ? cost[i][j] : pad,
    ),
  );

  const row_to_col = hungarian_square(square);
  const result = new Array<number>(n_prev).fill(-1);
  for (let i = 0; i < n_prev; i++) {
    const j = row_to_col[i];
    if (j >= 0 && j < n_curr) result[i] = j;
  }
  return result;
}

/** Cosine cost matrix between previous and current representative vectors. */
function cosine_cost_matrix(prev: number[][], curr: number[][]): number[][] {
  const n_prev = prev.length;
  const n_curr = curr.length;
  if (n_prev === 0 || n_curr === 0) {
    return Array.from({ length: n_prev }, () => new Array<number>(n_curr).fill(0));
  }
  const d = prev[0].length;
  return tf.tidy(() => {
    const combined = tf.tensor2d(
      [...prev, ...curr],
      [n_prev + n_curr, d],
      'float32',
    );
    const full = pairwise_distance_matrix(combined, 'cosine').arraySync() as number[][];
    const out: number[][] = new Array<number[]>(n_prev);
    for (let i = 0; i < n_prev; i++) {
      out[i] = full[i].slice(n_prev, n_prev + n_curr);
    }
    return out;
  });
}

/**
 * Tracks clusters between two consecutive snapshots.
 *
 * @param prev Representative vectors of the previous snapshot's clusters.
 * @param curr Representative vectors of the current snapshot's clusters.
 * @param options Tracking options (similarity threshold).
 * @param prev_state Lifeline state returned by the previous call; omit on the
 *   first frame to seed lifelines `0..n_prev-1`.
 *
 * Rectangular cases (differing cluster counts) are handled: extra clusters on
 * either side simply remain unmatched and surface as `EMERGE`/`DIE`.
 */
export function track_clusters(
  prev: number[][],
  curr: number[][],
  options: ClusterTrackingOptions = {},
  prev_state?: TrackingState,
): TrackingResult {
  const threshold = options.threshold ?? 0.5;
  const max_cost = 1 - threshold; // cosine cost ceiling for a candidate match
  const n_prev = prev.length;
  const n_curr = curr.length;

  const cost_matrix = cosine_cost_matrix(prev, curr);

  // Seed or adopt previous lifelines.
  const prev_lifelines =
    prev_state?.lifelines ?? Array.from({ length: n_prev }, (_v, i) => i);
  let next_id = prev_state?.next_lifeline_id ?? n_prev;

  // Optimal bipartite assignment (prev -> curr), then prune above threshold.
  const prev_to_curr = linear_sum_assignment(cost_matrix);

  // assignment[j] = prev matched to curr j (pruned), else -1.
  const assignment = new Array<number>(n_curr).fill(-1);
  for (let i = 0; i < n_prev; i++) {
    const j = prev_to_curr[i];
    if (j >= 0 && cost_matrix[i][j] <= max_cost) {
      assignment[j] = i;
    }
  }

  // Candidate valid edges (similarity >= threshold) for merge/split detection.
  const out_curr: number[][] = Array.from({ length: n_prev }, () => []);
  const in_prev: number[][] = Array.from({ length: n_curr }, () => []);
  for (let i = 0; i < n_prev; i++) {
    for (let j = 0; j < n_curr; j++) {
      if (cost_matrix[i][j] <= max_cost) {
        out_curr[i].push(j);
        in_prev[j].push(i);
      }
    }
  }

  // Lifelines for the current snapshot.
  const curr_lifelines = new Array<number>(n_curr).fill(-1);
  for (let j = 0; j < n_curr; j++) {
    const i = assignment[j];
    if (i >= 0) {
      curr_lifelines[j] = prev_lifelines[i];
    }
  }
  for (let j = 0; j < n_curr; j++) {
    if (curr_lifelines[j] === -1) curr_lifelines[j] = next_id++;
  }

  // Classify transitions.
  const transitions: ClusterTransition[] = [];

  for (let j = 0; j < n_curr; j++) {
    if (in_prev[j].length === 0) {
      transitions.push({ type: 'EMERGE', prev: [], curr: [j], lifeline_id: curr_lifelines[j] });
    } else if (in_prev[j].length >= 2) {
      transitions.push({
        type: 'MERGE',
        prev: [...in_prev[j]],
        curr: [j],
        lifeline_id: curr_lifelines[j],
      });
    }
  }

  for (let i = 0; i < n_prev; i++) {
    if (out_curr[i].length === 0) {
      transitions.push({
        type: 'DIE',
        prev: [i],
        curr: [],
        lifeline_id: prev_lifelines[i],
      });
    } else if (out_curr[i].length >= 2) {
      transitions.push({
        type: 'SPLIT',
        prev: [i],
        curr: [...out_curr[i]],
        lifeline_id: prev_lifelines[i],
      });
    }
  }

  // PERSIST: a clean one-to-one valid match.
  for (let j = 0; j < n_curr; j++) {
    if (in_prev[j].length === 1) {
      const i = in_prev[j][0];
      if (out_curr[i].length === 1) {
        transitions.push({
          type: 'PERSIST',
          prev: [i],
          curr: [j],
          lifeline_id: curr_lifelines[j],
        });
      }
    }
  }

  return {
    assignment,
    cost_matrix,
    transitions,
    state: { lifelines: curr_lifelines, next_lifeline_id: next_id },
  };
}
