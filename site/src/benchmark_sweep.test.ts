/// <reference types="jest" />
import { BENCH_SWEEP, N_MAX, N_MIN, summarize } from "./benchmark_sweep";

describe("BENCH_SWEEP", () => {
  test("is strictly ascending so points stream left-to-right on the chart", () => {
    for (let i = 1; i < BENCH_SWEEP.length; i++) {
      expect(BENCH_SWEEP[i]).toBeGreaterThan(BENCH_SWEEP[i - 1]);
    }
  });

  test("exposes its endpoints as the chart x-domain", () => {
    expect(N_MIN).toBe(BENCH_SWEEP[0]);
    expect(N_MAX).toBe(BENCH_SWEEP[BENCH_SWEEP.length - 1]);
  });
});

describe("summarize", () => {
  test("reports the peak speedup and its size over the shared sizes", () => {
    const cpu = new Map([
      [250, 5],
      [1000, 40],
      [4000, 800],
    ]);
    const accel = new Map([
      [250, { ms: 10, pts_per_sec: 25_000 }], // CPU wins small n
      [1000, { ms: 20, pts_per_sec: 50_000 }], // 2x
      [4000, { ms: 50, pts_per_sec: 80_000 }], // 16x — peak
    ]);
    const summary = summarize(cpu, accel);
    expect(summary.peak_speedup).toBeCloseTo(16);
    expect(summary.peak_speedup_n).toBe(4000);
  });

  test("takes throughput from the largest size both lanes reached", () => {
    const cpu = new Map([
      [250, 5],
      [1000, 40],
    ]);
    const accel = new Map([
      [250, { ms: 2, pts_per_sec: 125_000 }],
      [1000, { ms: 8, pts_per_sec: 125_000 }],
      // 4000 has no CPU counterpart (CPU lane retired), so it is excluded.
      [4000, { ms: 30, pts_per_sec: 133_000 }],
    ]);
    const summary = summarize(cpu, accel);
    expect(summary.throughput_pts_per_sec).toBe(125_000);
  });

  test("yields a zero summary when no size has both lanes", () => {
    const summary = summarize(
      new Map([[250, 5]]),
      new Map([[1000, { ms: 8, pts_per_sec: 1 }]]),
    );
    expect(summary.peak_speedup).toBe(0);
    expect(summary.throughput_pts_per_sec).toBe(0);
  });
});
