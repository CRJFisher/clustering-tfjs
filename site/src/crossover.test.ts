/// <reference types="jest" />
import { make_crossover_estimator } from "./crossover";

test("reports unknown before any race is recorded", () => {
  const estimator = make_crossover_estimator();
  expect(estimator.estimate()).toEqual({ kind: "unknown" });
});

test("reports below_range when every raced n is a GPU win", () => {
  const estimator = make_crossover_estimator();
  // GPU faster at both n's (gpu_ms < cpu_ms): the crossover lies below them.
  estimator.add_sample({ n: 2000, cpu_ms: 40, gpu_ms: 8 });
  estimator.add_sample({ n: 3000, cpu_ms: 90, gpu_ms: 10 });
  expect(estimator.estimate()).toEqual({ kind: "below_range" });
});

test("reports above_range when every raced n is a CPU win", () => {
  const estimator = make_crossover_estimator();
  // CPU faster at both n's (cpu_ms < gpu_ms): the crossover lies above them.
  estimator.add_sample({ n: 200, cpu_ms: 2, gpu_ms: 9 });
  estimator.add_sample({ n: 400, cpu_ms: 5, gpu_ms: 10 });
  expect(estimator.estimate()).toEqual({ kind: "above_range" });
});

test("interpolates the crossover between a CPU-win and a GPU-win bracket", () => {
  const estimator = make_crossover_estimator();
  // delta = gpu_ms − cpu_ms: +5 at n=1000 (CPU wins), −5 at n=2000 (GPU wins).
  // Symmetric magnitudes put the zero crossing at the midpoint, n=1500.
  estimator.add_sample({ n: 1000, cpu_ms: 10, gpu_ms: 15 });
  estimator.add_sample({ n: 2000, cpu_ms: 15, gpu_ms: 10 });
  const result = estimator.estimate();
  expect(result.kind).toBe("bracketed");
  if (result.kind === "bracketed") expect(result.n).toBeCloseTo(1500, 6);
});

test("interpolation weights the crossover toward the closer-to-even endpoint", () => {
  const estimator = make_crossover_estimator();
  // delta = +1 at n=1000, −9 at n=2000 → crossing sits near the low end.
  // n = 1000 + 1000 * 1 / (1 - (-9)) = 1100.
  estimator.add_sample({ n: 1000, cpu_ms: 10, gpu_ms: 11 });
  estimator.add_sample({ n: 2000, cpu_ms: 20, gpu_ms: 11 });
  const result = estimator.estimate();
  expect(result.kind).toBe("bracketed");
  if (result.kind === "bracketed") expect(result.n).toBeCloseTo(1100, 6);
});

test("takes the lowest-n sign change when noise produces a later flip", () => {
  const estimator = make_crossover_estimator();
  estimator.add_sample({ n: 500, cpu_ms: 5, gpu_ms: 10 }); // CPU wins
  estimator.add_sample({ n: 1500, cpu_ms: 20, gpu_ms: 12 }); // GPU wins (real crossing here)
  estimator.add_sample({ n: 3000, cpu_ms: 30, gpu_ms: 31 }); // noisy CPU win again
  const result = estimator.estimate();
  expect(result.kind).toBe("bracketed");
  // Crossing interpolated in [500, 1500], not the spurious [1500, 3000] flip.
  if (result.kind === "bracketed") {
    expect(result.n).toBeGreaterThan(500);
    expect(result.n).toBeLessThan(1500);
  }
});

test("re-racing the same n refines that point instead of duplicating it", () => {
  const estimator = make_crossover_estimator();
  estimator.add_sample({ n: 1000, cpu_ms: 10, gpu_ms: 15 });
  estimator.add_sample({ n: 2000, cpu_ms: 15, gpu_ms: 10 });
  // A fresh, GPU-winning measurement at n=1000 turns the low end into a GPU win
  // too, so no bracket remains and the crossover drops below the raced range.
  estimator.add_sample({ n: 1000, cpu_ms: 30, gpu_ms: 5 });
  expect(estimator.estimate()).toEqual({ kind: "below_range" });
});
