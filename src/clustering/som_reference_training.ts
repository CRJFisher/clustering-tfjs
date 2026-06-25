import type { SOMTopology, SOMNeighborhood } from './types';

/**
 * Numeric reference trainer that replicates MiniSom's `train_batch` exactly.
 *
 * `train_batch` is deterministic online-sequential training: for iteration
 * `t = 0 .. num_iteration - 1` it draws sample `data[t % n_samples]`, finds the
 * best-matching unit (BMU), and applies a single-sample update over the whole
 * grid `w += eta(t) * g(t) * (x - w)`. Learning rate and neighborhood radius
 * follow MiniSom's asymptotic decay `p(t) = p0 / (1 + t / (num_iteration / 2))`.
 *
 * The only stochastic input to MiniSom's `train_batch` is the initial weight
 * grid; injecting identical initial weights makes the run fully reproducible,
 * so this trainer matches MiniSom to floating-point precision.
 *
 * All math runs in plain JavaScript arrays (no tensors): grids are small,
 * training is inherently sequential, and tensor reductions reorder additions in
 * ways that break bit-for-bit parity. Internally the trainer works in MiniSom's
 * native `[grid_width][grid_height][n_features]` orientation (MiniSom indexes
 * weights as `[x][y]`); it transposes only at the API boundary so the public
 * inputs and outputs use the library's `[grid_height][grid_width][n_features]`
 * convention.
 *
 * This module is the SOM benchmarking reference path. It is intentionally
 * separate from the production `SOM` class (online mini-batch training), which
 * uses a different, GPU-friendly update rule. See `docs/som-benchmarking.md`.
 */

/** Vertical spacing between hexagonal rows: MiniSom's literal `Y_HEX_CONV_FACTOR`. */
const Y_HEX_CONV_FACTOR = 0.8660254037844387;

/** Absolute and relative tolerances for the hexagonal adjacency test (numpy `isclose` defaults). */
const ISCLOSE_ATOL = 1e-8;
const ISCLOSE_RTOL = 1e-5;

export interface MiniSomReferenceParams {
  /** MiniSom `x` dimension (number of columns). */
  grid_width: number;
  /** MiniSom `y` dimension (number of rows). */
  grid_height: number;
  topology: SOMTopology;
  neighborhood: SOMNeighborhood;
  learning_rate: number;
  /** Initial neighborhood radius `sigma0` (MiniSom `sigma`, the fixture `radius`). */
  sigma: number;
  /** Number of per-sample updates (MiniSom `num_iteration`, not epochs). */
  num_iteration: number;
}

export interface MiniSomReferenceResult {
  weights: number[][][];
  /** BMU grid coordinates per sample as `[row, col]`. */
  bmus: number[][];
  /** Flat cluster labels per sample (`row * grid_width + col`). */
  labels: number[];
  quantization_error: number;
  topographic_error: number;
  u_matrix: number[][];
}

/**
 * Euclidean coordinates of every neuron in the grid, in native `[width][height]`
 * layout. For rectangular topology these are the integer grid indices. For
 * hexagonal topology, odd rows (counting from the last row, matching MiniSom's
 * `_xx[::-2] -= 0.5`) are shifted by `-0.5` and rows are scaled vertically by
 * `Y_HEX_CONV_FACTOR`.
 */
function build_coordinate_grids(
  grid_width: number,
  grid_height: number,
  topology: SOMTopology,
): { xx: number[][]; yy: number[][] } {
  const xx: number[][] = [];
  const yy: number[][] = [];
  for (let a = 0; a < grid_width; a++) {
    const xx_row: number[] = [];
    const yy_row: number[] = [];
    for (let b = 0; b < grid_height; b++) {
      if (topology === 'hexagonal') {
        const offset = (grid_height - 1 - b) % 2 === 0 ? 0.5 : 0;
        xx_row.push(a - offset);
        yy_row.push(b * Y_HEX_CONV_FACTOR);
      } else {
        xx_row.push(a);
        yy_row.push(b);
      }
    }
    xx.push(xx_row);
    yy.push(yy_row);
  }
  return { xx, yy };
}

function squared_distance(sample: number[], weight: number[]): number {
  let sum = 0;
  for (let f = 0; f < sample.length; f++) {
    const d = sample[f] - weight[f];
    sum += d * d;
  }
  return sum;
}

/**
 * Best-matching unit for a sample, returned as `[col, row]` (MiniSom `(x, y)`).
 * Iterates width-major then height (C-order over MiniSom's `(width, height)`
 * activation map) and keeps the first minimum, matching numpy `argmin`.
 */
function find_bmu(
  sample: number[],
  weights_native: number[][][],
  grid_width: number,
  grid_height: number,
): [number, number] {
  let best_a = 0;
  let best_b = 0;
  let best_dist = Infinity;
  for (let a = 0; a < grid_width; a++) {
    for (let b = 0; b < grid_height; b++) {
      const dist = squared_distance(sample, weights_native[a][b]);
      if (dist < best_dist) {
        best_dist = dist;
        best_a = a;
        best_b = b;
      }
    }
  }
  return [best_a, best_b];
}

/** MiniSom asymptotic decay: `value0 / (1 + t / (num_iteration / 2))`. */
function asymptotic_decay(value0: number, t: number, num_iteration: number): number {
  return value0 / (1 + t / (num_iteration / 2));
}

/**
 * Neighborhood influence `g[col][row]` centered on the BMU `(cx, cy)`.
 * Replicates MiniSom's three neighborhood functions exactly.
 */
function neighborhood_influence(
  cx: number,
  cy: number,
  sigma: number,
  neighborhood: SOMNeighborhood,
  grid_width: number,
  grid_height: number,
  xx: number[][],
  yy: number[][],
): number[][] {
  const g: number[][] = [];

  if (neighborhood === 'bubble') {
    // Separable open-interval box on integer grid indices (MiniSom `_bubble`):
    // 1 where `c - sigma < index < c + sigma`, strict on both sides.
    for (let a = 0; a < grid_width; a++) {
      const ax = a > cx - sigma && a < cx + sigma ? 1 : 0;
      const g_row: number[] = [];
      for (let b = 0; b < grid_height; b++) {
        const ay = b > cy - sigma && b < cy + sigma ? 1 : 0;
        g_row.push(ax * ay);
      }
      g.push(g_row);
    }
    return g;
  }

  const x_c = xx[cx][cy];
  const y_c = yy[cx][cy];
  const d = 2 * sigma * sigma;

  for (let a = 0; a < grid_width; a++) {
    const g_row: number[] = [];
    for (let b = 0; b < grid_height; b++) {
      const dx = xx[a][b] - x_c;
      const dy = yy[a][b] - y_c;
      const p = dx * dx + dy * dy;
      if (neighborhood === 'gaussian') {
        // Separable product of 1-D gaussians == exp(-p / 2 sigma^2).
        g_row.push(Math.exp(-p / d));
      } else {
        // mexican_hat (Ricker): exp(-p / d) * (1 - 2 p / d), d = 2 sigma^2.
        g_row.push(Math.exp(-p / d) * (1 - (2 / d) * p));
      }
    }
    g.push(g_row);
  }
  return g;
}

function transpose_to_native(weights: number[][][]): number[][][] {
  const grid_height = weights.length;
  const grid_width = weights[0].length;
  const native: number[][][] = [];
  for (let a = 0; a < grid_width; a++) {
    const col: number[][] = [];
    for (let b = 0; b < grid_height; b++) {
      col.push(weights[b][a].slice());
    }
    native.push(col);
  }
  return native;
}

function transpose_from_native(weights_native: number[][][]): number[][][] {
  const grid_width = weights_native.length;
  const grid_height = weights_native[0].length;
  const weights: number[][][] = [];
  for (let b = 0; b < grid_height; b++) {
    const row: number[][] = [];
    for (let a = 0; a < grid_width; a++) {
      row.push(weights_native[a][b].slice());
    }
    weights.push(row);
  }
  return weights;
}

function quantization_error(
  data: number[][],
  weights_native: number[][][],
  grid_width: number,
  grid_height: number,
): number {
  let total = 0;
  for (const sample of data) {
    const [a, b] = find_bmu(sample, weights_native, grid_width, grid_height);
    total += Math.sqrt(squared_distance(sample, weights_native[a][b]));
  }
  return total / data.length;
}

/** Flat indices of the two nearest neurons for a sample, C-order over (width, height). */
function two_nearest_flat_indices(
  sample: number[],
  weights_native: number[][][],
  grid_width: number,
  grid_height: number,
): [number, number] {
  const distances: { dist: number; index: number }[] = [];
  for (let a = 0; a < grid_width; a++) {
    for (let b = 0; b < grid_height; b++) {
      distances.push({ dist: squared_distance(sample, weights_native[a][b]), index: a * grid_height + b });
    }
  }
  distances.sort((u, v) => (u.dist === v.dist ? u.index - v.index : u.dist - v.dist));
  return [distances[0].index, distances[1].index];
}

/**
 * Topographic error: fraction of samples whose two nearest neurons are not
 * adjacent on the grid. Replicates MiniSom's topology-specific definitions.
 */
function topographic_error(
  data: number[][],
  weights_native: number[][][],
  grid_width: number,
  grid_height: number,
  topology: SOMTopology,
  xx: number[][],
  yy: number[][],
): number {
  let errors = 0;
  for (const sample of data) {
    const [first, second] = two_nearest_flat_indices(sample, weights_native, grid_width, grid_height);
    const a1 = Math.floor(first / grid_height);
    const b1 = first % grid_height;
    const a2 = Math.floor(second / grid_height);
    const b2 = second % grid_height;

    if (topology === 'hexagonal') {
      // MiniSom: neurons are adjacent iff their Euclidean hex coordinates are unit distance.
      const dx = xx[a1][b1] - xx[a2][b2];
      const dy = yy[a1][b1] - yy[a2][b2];
      const dist = Math.sqrt(dx * dx + dy * dy);
      // numpy isclose(1, dist): |dist - 1| <= atol + rtol * |dist|.
      if (Math.abs(dist - 1) > ISCLOSE_ATOL + ISCLOSE_RTOL * Math.abs(dist)) {
        errors++;
      }
    } else {
      // MiniSom: adjacent iff grid-index distance <= sqrt(2); error threshold is 1.42.
      const da = a1 - a2;
      const db = b1 - b2;
      if (Math.sqrt(da * da + db * db) > 1.42) {
        errors++;
      }
    }
  }
  return errors / data.length;
}

/**
 * Normalized inter-neuron distance map (MiniSom `distance_map(scaling='sum')`),
 * returned in `[grid_height][grid_width]` orientation. Each cell is the summed
 * Euclidean distance to its in-bounds neighbors, divided by the grid maximum.
 */
function distance_map(
  weights_native: number[][][],
  grid_width: number,
  grid_height: number,
  topology: SOMTopology,
): number[][] {
  // Neighbor offsets in MiniSom's (x=width, y=height) index space.
  const ii_rect = [0, -1, -1, -1, 0, 1, 1, 1];
  const jj_rect = [-1, -1, 0, 1, 1, 1, 0, -1];
  const ii_hex = [
    [1, 1, 1, 0, -1, 0],
    [0, 1, 0, -1, -1, -1],
  ];
  const jj_hex = [
    [1, 0, -1, -1, 0, 1],
    [1, 0, -1, -1, 0, 1],
  ];

  const summed: number[][] = [];
  let max_value = 0;
  for (let a = 0; a < grid_width; a++) {
    const row: number[] = [];
    for (let b = 0; b < grid_height; b++) {
      let ii: number[];
      let jj: number[];
      if (topology === 'hexagonal') {
        const e = b % 2 === 0 ? 1 : 0;
        ii = ii_hex[e];
        jj = jj_hex[e];
      } else {
        ii = ii_rect;
        jj = jj_rect;
      }
      let sum = 0;
      for (let k = 0; k < ii.length; k++) {
        const na = a + ii[k];
        const nb = b + jj[k];
        if (na >= 0 && na < grid_width && nb >= 0 && nb < grid_height) {
          sum += Math.sqrt(squared_distance(weights_native[a][b], weights_native[na][nb]));
        }
      }
      if (sum > max_value) {
        max_value = sum;
      }
      row.push(sum);
    }
    summed.push(row);
  }

  const u_matrix: number[][] = [];
  for (let b = 0; b < grid_height; b++) {
    const row: number[] = [];
    for (let a = 0; a < grid_width; a++) {
      row.push(max_value > 0 ? summed[a][b] / max_value : summed[a][b]);
    }
    u_matrix.push(row);
  }
  return u_matrix;
}

export function train_minisom_reference(
  data: number[][],
  initial_weights: number[][][],
  params: MiniSomReferenceParams,
): MiniSomReferenceResult {
  const { grid_width, grid_height, topology, neighborhood, learning_rate, sigma, num_iteration } = params;
  const n_samples = data.length;

  const weights_native = transpose_to_native(initial_weights);
  const { xx, yy } = build_coordinate_grids(grid_width, grid_height, topology);

  for (let t = 0; t < num_iteration; t++) {
    const sample = data[t % n_samples];
    const [cx, cy] = find_bmu(sample, weights_native, grid_width, grid_height);
    const eta = asymptotic_decay(learning_rate, t, num_iteration);
    const sig = asymptotic_decay(sigma, t, num_iteration);
    const g = neighborhood_influence(cx, cy, sig, neighborhood, grid_width, grid_height, xx, yy);

    for (let a = 0; a < grid_width; a++) {
      for (let b = 0; b < grid_height; b++) {
        const factor = eta * g[a][b];
        if (factor === 0) {
          continue;
        }
        const weight = weights_native[a][b];
        for (let f = 0; f < weight.length; f++) {
          weight[f] += factor * (sample[f] - weight[f]);
        }
      }
    }
  }

  const bmus: number[][] = [];
  const labels: number[] = [];
  for (const sample of data) {
    const [a, b] = find_bmu(sample, weights_native, grid_width, grid_height);
    bmus.push([b, a]); // [row, col]
    labels.push(b * grid_width + a);
  }

  return {
    weights: transpose_from_native(weights_native),
    bmus,
    labels,
    quantization_error: quantization_error(data, weights_native, grid_width, grid_height),
    topographic_error: topographic_error(data, weights_native, grid_width, grid_height, topology, xx, yy),
    u_matrix: distance_map(weights_native, grid_width, grid_height, topology),
  };
}
