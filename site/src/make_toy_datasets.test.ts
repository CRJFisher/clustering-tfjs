/// <reference types="jest" />
import { make_toy_dataset, type ToyDatasetId } from "./make_toy_datasets";

const ALL_IDS: ToyDatasetId[] = ["moons", "circles", "blobs", "aniso", "none"];
const N_SAMPLES = 300;

function radius(data: Float32Array, i: number): number {
  return Math.hypot(data[i * 2], data[i * 2 + 1]);
}

function distinct_labels(labels: Int32Array): Set<number> {
  return new Set(Array.from(labels));
}

test.each(ALL_IDS)("%s has the right shape and types", (id) => {
  const ds = make_toy_dataset(id);
  expect(ds.n_samples).toBe(N_SAMPLES);
  expect(ds.data).toBeInstanceOf(Float32Array);
  expect(ds.labels).toBeInstanceOf(Int32Array);
  expect(ds.data.length).toBe(N_SAMPLES * 2);
  expect(ds.labels.length).toBe(N_SAMPLES);
});

test.each(ALL_IDS)("%s is byte-for-byte reproducible", (id) => {
  const a = make_toy_dataset(id);
  const b = make_toy_dataset(id);
  expect(Array.from(b.data)).toEqual(Array.from(a.data));
  expect(Array.from(b.labels)).toEqual(Array.from(a.labels));
});

test.each(ALL_IDS)("%s contains no NaN or Infinity", (id) => {
  const ds = make_toy_dataset(id);
  for (const value of ds.data) expect(Number.isFinite(value)).toBe(true);
});

test.each(ALL_IDS)("%s is standardized to ~zero mean, ~unit variance", (id) => {
  const ds = make_toy_dataset(id);
  for (let col = 0; col < 2; col++) {
    let mean = 0;
    for (let i = 0; i < N_SAMPLES; i++) mean += ds.data[i * 2 + col];
    mean /= N_SAMPLES;
    let variance = 0;
    for (let i = 0; i < N_SAMPLES; i++) {
      const d = ds.data[i * 2 + col] - mean;
      variance += d * d;
    }
    variance /= N_SAMPLES;
    expect(Math.abs(mean)).toBeLessThan(1e-3);
    expect(Math.abs(variance - 1)).toBeLessThan(1e-3);
  }
});

test("moons and circles carry exactly two ground-truth labels", () => {
  expect(distinct_labels(make_toy_dataset("moons").labels)).toEqual(
    new Set([0, 1]),
  );
  expect(distinct_labels(make_toy_dataset("circles").labels)).toEqual(
    new Set([0, 1]),
  );
});

test("blobs and aniso carry exactly three ground-truth labels", () => {
  expect(distinct_labels(make_toy_dataset("blobs").labels)).toEqual(
    new Set([0, 1, 2]),
  );
  expect(distinct_labels(make_toy_dataset("aniso").labels)).toEqual(
    new Set([0, 1, 2]),
  );
});

test("no-structure has a single degenerate label", () => {
  expect(distinct_labels(make_toy_dataset("none").labels)).toEqual(
    new Set([0]),
  );
});

test("circles inner ring sits inside the outer ring", () => {
  const ds = make_toy_dataset("circles");
  let inner = 0;
  let outer = 0;
  let inner_n = 0;
  let outer_n = 0;
  for (let i = 0; i < ds.n_samples; i++) {
    if (ds.labels[i] === 1) {
      inner += radius(ds.data, i);
      inner_n++;
    } else {
      outer += radius(ds.data, i);
      outer_n++;
    }
  }
  expect(inner / inner_n).toBeLessThan(outer / outer_n);
});

test("aniso transform introduces cross-column correlation absent from blobs", () => {
  // The shear rotates the principal axes off the coordinate axes, so |cov_xy| is
  // meaningfully non-zero — the assertion that proves the transform was applied.
  function abs_cross_cov(id: ToyDatasetId): number {
    const ds = make_toy_dataset(id);
    let cov = 0;
    for (let i = 0; i < ds.n_samples; i++) {
      cov += ds.data[i * 2] * ds.data[i * 2 + 1];
    }
    return Math.abs(cov / ds.n_samples);
  }
  expect(abs_cross_cov("aniso")).toBeGreaterThan(abs_cross_cov("blobs") + 0.1);
});

test("different datasets are distinct", () => {
  const moons = make_toy_dataset("moons");
  const circles = make_toy_dataset("circles");
  expect(Array.from(moons.data)).not.toEqual(Array.from(circles.data));
});
