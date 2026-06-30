// A dependency-free, seeded Gaussian-blob generator that runs on the main
// thread. The library's own make_blobs returns tfjs tensors and needs an
// initialized engine; the main thread deliberately never initializes tfjs (all
// compute lives in the workers), and a pure-JS generator also guarantees the
// two lanes receive byte-identical float32 input — the fairness protocol's
// first rule.

export interface MakeBlobsOptions {
  n_samples: number;
  n_features: number;
  centers: number;
  cluster_std: number;
  // Fixed seed → the benchmark is reproducible across visitors.
  random_state: number;
}

export interface MakeBlobsResult {
  // Row-major n_samples × n_features, float32.
  data: Float32Array;
  // Ground-truth cluster index per row, for an optional "same result" overlay.
  labels: Int32Array;
}

// mulberry32: a tiny, fast, well-distributed 32-bit PRNG. Deterministic from a
// single integer seed so every visitor with the same n/seed sees the same data.
// Exported so the toy-dataset generators draw from the identical PRNG character
// as the blobs here — one seeded stream, no forked randomness across the site.
export function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// Box–Muller transforms two uniforms into a standard-normal sample.
export function next_gaussian(rand: () => number): number {
  let u = 0;
  let v = 0;
  // Avoid log(0): resample the (vanishingly rare) exact-zero draw.
  while (u === 0) u = rand();
  while (v === 0) v = rand();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

export function make_blobs_js(options: MakeBlobsOptions): MakeBlobsResult {
  const { n_samples, n_features, centers, cluster_std, random_state } = options;

  const rand = mulberry32(random_state);

  // Cluster centers spread across a fixed box so blobs separate cleanly and the
  // RBF affinity has real structure to find.
  const center_coords = new Float32Array(centers * n_features);
  const CENTER_SPREAD = 10;
  for (let c = 0; c < centers; c++) {
    for (let f = 0; f < n_features; f++) {
      center_coords[c * n_features + f] = (rand() - 0.5) * 2 * CENTER_SPREAD;
    }
  }

  const data = new Float32Array(n_samples * n_features);
  const labels = new Int32Array(n_samples);

  for (let i = 0; i < n_samples; i++) {
    const c = i % centers;
    labels[i] = c;
    const center_offset = c * n_features;
    const row_offset = i * n_features;
    for (let f = 0; f < n_features; f++) {
      data[row_offset + f] =
        center_coords[center_offset + f] + next_gaussian(rand) * cluster_std;
    }
  }

  return { data, labels };
}
