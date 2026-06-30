import { mulberry32, next_gaussian } from "./make_blobs_js";

// The five "toy datasets" the clustering grid runs every algorithm against,
// generated in plain JS on the main thread. The library only ships make_blobs;
// the moons / circles / anisotropic / no-structure shapes have no generator, so
// they live here — seeded from one mulberry32 stream (shared with make_blobs_js)
// so every visitor sees the byte-identical grid. Each is a deterministic
// reproduction of a classic clustering-benchmark shape.

export type ToyDatasetId =
  | "moons"
  | "circles"
  | "blobs"
  | "aniso"
  | "none";

export interface ToyDataset {
  // Row-major n_samples × 2, float32, standardized (zero mean, unit variance per
  // column) so every cell's distance/affinity params are calibrated to one scale.
  data: Float32Array;
  n_samples: number;
  // Ground-truth label per row. Degenerate (all 0) for no-structure, where there
  // is nothing to recover — used only by tests, never to colour the grid (cells
  // are coloured by the labels the library predicts).
  labels: Int32Array;
}

// Every grid dataset is this size: large enough to read as the iconic shapes,
// small enough that 25 sequential fits in one worker stay snappy.
const N_SAMPLES = 300;

// Re-centre and rescale each column to zero mean and unit variance. Standardizing
// every dataset before clustering is what lets one fixed gamma / n_neighbors /
// min_cluster_size in grid_config serve all five rows without per-row recalibration.
function standardize(data: Float32Array, n_samples: number): void {
  for (let col = 0; col < 2; col++) {
    let mean = 0;
    for (let i = 0; i < n_samples; i++) mean += data[i * 2 + col];
    mean /= n_samples;
    let variance = 0;
    for (let i = 0; i < n_samples; i++) {
      const d = data[i * 2 + col] - mean;
      variance += d * d;
    }
    variance /= n_samples;
    const std = Math.sqrt(variance) || 1;
    for (let i = 0; i < n_samples; i++) {
      data[i * 2 + col] = (data[i * 2 + col] - mean) / std;
    }
  }
}

// Two interleaving half-circles ("two moons"). The outer moon is the upper unit
// semicircle; the inner moon is its point-reflection shifted right and down so the
// two arcs interlock — the canonical "can your algorithm follow a curved manifold?"
// test. Row order is irrelevant to clustering, so no shuffle is applied.
function make_moons(noise: number, random_state: number): ToyDataset {
  const rand = mulberry32(random_state);
  const n_out = N_SAMPLES >> 1;
  const n_in = N_SAMPLES - n_out;
  const data = new Float32Array(N_SAMPLES * 2);
  const labels = new Int32Array(N_SAMPLES);

  for (let i = 0; i < n_out; i++) {
    const t = n_out > 1 ? (Math.PI * i) / (n_out - 1) : 0;
    data[i * 2] = Math.cos(t);
    data[i * 2 + 1] = Math.sin(t);
    labels[i] = 0;
  }
  for (let i = 0; i < n_in; i++) {
    const t = n_in > 1 ? (Math.PI * i) / (n_in - 1) : 0;
    const row = n_out + i;
    data[row * 2] = 1 - Math.cos(t);
    data[row * 2 + 1] = 1 - Math.sin(t) - 0.5;
    labels[row] = 1;
  }

  add_gaussian_noise(data, rand, noise);
  standardize(data, N_SAMPLES);
  return { data, n_samples: N_SAMPLES, labels };
}

// Two concentric circles. The inner circle's radius is `factor` times the outer;
// a clustering that keys on Euclidean compactness cannot separate them, which is
// the whole point of the row.
function make_circles(
  noise: number,
  factor: number,
  random_state: number,
): ToyDataset {
  const rand = mulberry32(random_state);
  const n_out = N_SAMPLES >> 1;
  const n_in = N_SAMPLES - n_out;
  const data = new Float32Array(N_SAMPLES * 2);
  const labels = new Int32Array(N_SAMPLES);

  for (let i = 0; i < n_out; i++) {
    const t = (2 * Math.PI * i) / n_out;
    data[i * 2] = Math.cos(t);
    data[i * 2 + 1] = Math.sin(t);
    labels[i] = 0;
  }
  for (let i = 0; i < n_in; i++) {
    const t = (2 * Math.PI * i) / n_in;
    const row = n_out + i;
    data[row * 2] = factor * Math.cos(t);
    data[row * 2 + 1] = factor * Math.sin(t);
    labels[row] = 1;
  }

  add_gaussian_noise(data, rand, noise);
  standardize(data, N_SAMPLES);
  return { data, n_samples: N_SAMPLES, labels };
}

// Three fixed, well-separated isotropic Gaussian centres. Fixed (not seed-random)
// centres keep the classic three-blob look stable across machines while the
// per-point jitter still comes from the seeded stream.
const BLOB_CENTERS: ReadonlyArray<readonly [number, number]> = [
  [-5, 0],
  [0, 5],
  [5, 0],
];

function fill_iso_blobs(
  data: Float32Array,
  labels: Int32Array,
  cluster_std: number,
  rand: () => number,
): void {
  for (let i = 0; i < N_SAMPLES; i++) {
    const c = i % BLOB_CENTERS.length;
    labels[i] = c;
    data[i * 2] = BLOB_CENTERS[c][0] + next_gaussian(rand) * cluster_std;
    data[i * 2 + 1] = BLOB_CENTERS[c][1] + next_gaussian(rand) * cluster_std;
  }
}

function make_blobs(cluster_std: number, random_state: number): ToyDataset {
  const rand = mulberry32(random_state);
  const data = new Float32Array(N_SAMPLES * 2);
  const labels = new Int32Array(N_SAMPLES);
  fill_iso_blobs(data, labels, cluster_std, rand);
  standardize(data, N_SAMPLES);
  return { data, n_samples: N_SAMPLES, labels };
}

// The anisotropic transform: isotropic blobs run through a fixed shear so the
// clusters become stretched, correlated ellipses. K-Means' spherical assumption
// breaks here while connectivity-aware methods cope — that contrast is the row's
// purpose. The matrix is applied as the row-vector product `[x, y] · T`.
const ANISO_TRANSFORM: readonly [number, number, number, number] = [
  0.6, -0.6, -0.4, 0.8,
];

function make_aniso(cluster_std: number, random_state: number): ToyDataset {
  const rand = mulberry32(random_state);
  const data = new Float32Array(N_SAMPLES * 2);
  const labels = new Int32Array(N_SAMPLES);
  fill_iso_blobs(data, labels, cluster_std, rand);

  const [a, b, c, d] = ANISO_TRANSFORM;
  for (let i = 0; i < N_SAMPLES; i++) {
    const x = data[i * 2];
    const y = data[i * 2 + 1];
    data[i * 2] = x * a + y * c;
    data[i * 2 + 1] = x * b + y * d;
  }

  standardize(data, N_SAMPLES);
  return { data, n_samples: N_SAMPLES, labels };
}

// Uniform points in the unit square: no clusters exist. The honest outcome the
// grid shows is that density methods report all-noise while partitioning methods
// are forced to invent an arbitrary split.
function make_no_structure(random_state: number): ToyDataset {
  const rand = mulberry32(random_state);
  const data = new Float32Array(N_SAMPLES * 2);
  const labels = new Int32Array(N_SAMPLES);
  for (let i = 0; i < N_SAMPLES; i++) {
    data[i * 2] = rand();
    data[i * 2 + 1] = rand();
  }
  standardize(data, N_SAMPLES);
  return { data, n_samples: N_SAMPLES, labels };
}

// Drawn AFTER all structured coordinates are placed, as a final pass over the
// whole array. Skipped entirely when noise is 0 so a clean shape keeps the PRNG
// stream tidy and never risks a vanishing 0×gaussian draw.
function add_gaussian_noise(
  data: Float32Array,
  rand: () => number,
  noise: number,
): void {
  if (noise === 0) return;
  for (let i = 0; i < data.length; i++) {
    data[i] += next_gaussian(rand) * noise;
  }
}

// Fixed per-dataset generation seeds and parameters — the single place the grid's
// shapes are pinned, so the whole 5×5 matrix is reproducible from this module.
export function make_toy_dataset(id: ToyDatasetId): ToyDataset {
  switch (id) {
    case "moons":
      return make_moons(0.08, 10);
    case "circles":
      return make_circles(0.05, 0.5, 20);
    case "blobs":
      return make_blobs(0.7, 30);
    case "aniso":
      return make_aniso(0.7, 40);
    case "none":
      return make_no_structure(50);
  }
}
