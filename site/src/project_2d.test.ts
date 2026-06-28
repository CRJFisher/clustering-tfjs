/// <reference types="jest" />
import { make_blobs_js } from "./make_blobs_js";
import { project_2d_pca } from "./project_2d";

const N_SAMPLES = 600;
const N_FEATURES = 32;
const CENTERS = 4;

function make_dataset(): { data: Float32Array; labels: Int32Array } {
  return make_blobs_js({
    n_samples: N_SAMPLES,
    n_features: N_FEATURES,
    centers: CENTERS,
    cluster_std: 1.5,
    random_state: 42,
  });
}

function variance(values: Float32Array): number {
  let mean = 0;
  for (let i = 0; i < values.length; i++) mean += values[i];
  mean /= values.length;
  let sum = 0;
  for (let i = 0; i < values.length; i++) sum += (values[i] - mean) ** 2;
  return sum / values.length;
}

test("projects every row to a 2-D coordinate", () => {
  const { data } = make_dataset();
  const { x, y } = project_2d_pca(data, N_SAMPLES, N_FEATURES);
  expect(x.length).toBe(N_SAMPLES);
  expect(y.length).toBe(N_SAMPLES);
});

test("is byte-for-byte reproducible so a shared permalink reproduces the scatter", () => {
  const { data } = make_dataset();
  const first = project_2d_pca(data, N_SAMPLES, N_FEATURES);
  const second = project_2d_pca(data, N_SAMPLES, N_FEATURES);
  expect(Array.from(second.x)).toEqual(Array.from(first.x));
  expect(Array.from(second.y)).toEqual(Array.from(first.y));
});

test("centers the projection at the origin", () => {
  const { data } = make_dataset();
  const { x, y } = project_2d_pca(data, N_SAMPLES, N_FEATURES);
  const mean_x = x.reduce((a, b) => a + b, 0) / x.length;
  const mean_y = y.reduce((a, b) => a + b, 0) / y.length;
  expect(Math.abs(mean_x)).toBeLessThan(1e-3);
  expect(Math.abs(mean_y)).toBeLessThan(1e-3);
});

test("orders components by variance: the first axis captures at least the second", () => {
  const { data } = make_dataset();
  const { x, y } = project_2d_pca(data, N_SAMPLES, N_FEATURES);
  expect(variance(x)).toBeGreaterThanOrEqual(variance(y));
});

test("separates the blobs: between-cluster spread dominates within-cluster spread", () => {
  const { data, labels } = make_dataset();
  const { x, y } = project_2d_pca(data, N_SAMPLES, N_FEATURES);

  const sum_x = new Float64Array(CENTERS);
  const sum_y = new Float64Array(CENTERS);
  const count = new Int32Array(CENTERS);
  for (let i = 0; i < N_SAMPLES; i++) {
    const c = labels[i];
    sum_x[c] += x[i];
    sum_y[c] += y[i];
    count[c] += 1;
  }
  const centroid_x = new Float64Array(CENTERS);
  const centroid_y = new Float64Array(CENTERS);
  for (let c = 0; c < CENTERS; c++) {
    centroid_x[c] = sum_x[c] / count[c];
    centroid_y[c] = sum_y[c] / count[c];
  }

  let within = 0;
  for (let i = 0; i < N_SAMPLES; i++) {
    const c = labels[i];
    within += (x[i] - centroid_x[c]) ** 2 + (y[i] - centroid_y[c]) ** 2;
  }
  const within_rms = Math.sqrt(within / N_SAMPLES);

  let min_between = Infinity;
  for (let a = 0; a < CENTERS; a++) {
    for (let b = a + 1; b < CENTERS; b++) {
      const distance = Math.hypot(
        centroid_x[a] - centroid_x[b],
        centroid_y[a] - centroid_y[b],
      );
      if (distance < min_between) min_between = distance;
    }
  }

  expect(min_between).toBeGreaterThan(within_rms);
});
