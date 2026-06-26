// Deterministic 2-D PCA for the scatter panels. The main thread never
// initializes tfjs (all compute lives in the workers), so this is plain JS over
// the seeded Float32Array. PCA — not two raw feature axes — because the blobs'
// centers sit at random positions across 32 dimensions, so two arbitrary axes
// rarely separate them; the top two principal components put the between-cluster
// variance on screen. It is the standard, defensible way to show high-D
// structure in 2-D, and it never bakes the ground-truth centers into the axes.

export interface Projection2d {
  // 2-D coordinates per row in principal-component units (not pixels — the
  // renderer owns the fit to the canvas box).
  x: Float32Array;
  y: Float32Array;
}

// A fixed iteration count, not a convergence tolerance: a tolerance stop can
// vary by a step across float rounding between machines, which would make a
// shared permalink reproduce a subtly different scatter. Blobs have a wide
// spectral gap, so 128 steps is far past convergence.
const POWER_ITERATIONS = 128;

// A fixed, varied start vector. Fixed → the same dataset projects identically on
// every machine. Varied (not all-ones) → an all-ones start is degenerate when
// the dominant eigenvector is orthogonal to it; a spread of values guarantees a
// non-zero projection onto the top component.
function seeded_start(d: number): Float64Array {
  const v = new Float64Array(d);
  let state = 0x9e3779b9;
  for (let j = 0; j < d; j++) {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    v[j] = state / 0xffffffff - 0.5;
  }
  return v;
}

function normalize(v: Float64Array): void {
  let norm = 0;
  for (let i = 0; i < v.length; i++) norm += v[i] * v[i];
  norm = Math.sqrt(norm);
  for (let i = 0; i < v.length; i++) v[i] /= norm;
}

function matvec(
  cov: Float64Array,
  v: Float64Array,
  d: number,
  out: Float64Array,
): void {
  for (let i = 0; i < d; i++) {
    let sum = 0;
    const row = i * d;
    for (let j = 0; j < d; j++) sum += cov[row + j] * v[j];
    out[i] = sum;
  }
}

// Dominant eigenvector of a symmetric d×d matrix by power iteration. When
// `orthogonal_to` is given it is projected out of the iterate every step, so
// (cov being symmetric, hence its eigenvectors orthogonal) the iteration
// converges to the SECOND eigenvector after the first is known. Re-projecting
// each step — rather than deflating the matrix once — keeps round-off from
// leaking the iterate back toward the first component.
function dominant_eigenvector(
  cov: Float64Array,
  d: number,
  orthogonal_to?: Float64Array,
): Float64Array {
  const v = seeded_start(d);
  normalize(v);
  const next = new Float64Array(d);
  for (let iter = 0; iter < POWER_ITERATIONS; iter++) {
    matvec(cov, v, d, next);
    if (orthogonal_to) {
      let dot = 0;
      for (let j = 0; j < d; j++) dot += next[j] * orthogonal_to[j];
      for (let j = 0; j < d; j++) next[j] -= dot * orthogonal_to[j];
    }
    normalize(next);
    v.set(next);
  }
  // Pin the sign so the largest-magnitude loading is positive. Eigenvectors are
  // sign-ambiguous; without this the scatter could randomly mirror between runs.
  let peak = 0;
  for (let j = 0; j < d; j++) {
    if (Math.abs(v[j]) > Math.abs(v[peak])) peak = j;
  }
  if (v[peak] < 0) {
    for (let j = 0; j < d; j++) v[j] = -v[j];
  }
  return v;
}

export function project_2d_pca(
  data: Float32Array,
  n_samples: number,
  n_features: number,
): Projection2d {
  const d = n_features;

  // Float64 accumulators throughout, narrowing only at the final store: float32
  // summation of 2000 rows drifts enough to make the covariance — and therefore
  // the eigenvectors — differ across machines, breaking determinism.
  const mean = new Float64Array(d);
  for (let i = 0; i < n_samples; i++) {
    const row = i * d;
    for (let j = 0; j < d; j++) mean[j] += data[row + j];
  }
  for (let j = 0; j < d; j++) mean[j] /= n_samples;

  // Covariance is symmetric, so accumulate only the upper triangle then mirror.
  const cov = new Float64Array(d * d);
  const centered = new Float64Array(d);
  for (let i = 0; i < n_samples; i++) {
    const row = i * d;
    for (let j = 0; j < d; j++) centered[j] = data[row + j] - mean[j];
    for (let a = 0; a < d; a++) {
      const ca = centered[a];
      const dst = a * d;
      for (let b = a; b < d; b++) cov[dst + b] += ca * centered[b];
    }
  }
  for (let a = 0; a < d; a++) {
    for (let b = a; b < d; b++) {
      const value = cov[a * d + b] / n_samples;
      cov[a * d + b] = value;
      cov[b * d + a] = value;
    }
  }

  const v1 = dominant_eigenvector(cov, d);
  const v2 = dominant_eigenvector(cov, d, v1);

  const x = new Float32Array(n_samples);
  const y = new Float32Array(n_samples);
  for (let i = 0; i < n_samples; i++) {
    const row = i * d;
    let px = 0;
    let py = 0;
    for (let j = 0; j < d; j++) {
      const value = data[row + j] - mean[j];
      px += value * v1[j];
      py += value * v2[j];
    }
    x[i] = px;
    y[i] = py;
  }
  return { x, y };
}
