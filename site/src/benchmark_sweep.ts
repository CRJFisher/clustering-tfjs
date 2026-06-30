import type { SeriesId } from "./benchmark_protocol";

// The pure, worker-free core of the benchmark: the sample-size sweep, the run
// config, and the closing-summary math. Kept separate from benchmark.ts (which
// spawns workers via import.meta.url) so this can be unit-tested under ts-jest's
// CommonJS target without tripping over import.meta.

// Geometric sweep: log-spaced sizes so the O(n²·d) scaling law reads as a straight
// line on the log–log chart. The top of the range pushes past the old dense-matrix
// ceiling; a size that OOMs is dropped per-lane (see run_benchmark) rather than
// aborting the sweep, so the chart still shows every size the device could handle.
export const BENCH_SWEEP: number[] = [250, 500, 1000, 2000, 4000, 8000];

// The smallest and largest swept sizes — the chart's x-domain.
export const N_MIN = BENCH_SWEEP[0];
export const N_MAX = BENCH_SWEEP[BENCH_SWEEP.length - 1];

export interface BenchmarkConfig {
  n_features: number;
  centers: number;
  cluster_std: number;
  random_state: number;
  gamma: number;
  warmups: number;
  reps: number;
}

export const DEFAULT_BENCHMARK_CONFIG: BenchmarkConfig = {
  // d=32 keeps the affinity O(n²·d) compute-bound where the accelerated win is
  // real; the fixed seed makes every visitor's curve reproducible.
  n_features: 32,
  centers: 4,
  cluster_std: 1.5,
  random_state: 42,
  // 1/n_features pinned explicitly so every size computes the identical kernel.
  gamma: 1 / 32,
  warmups: 3,
  reps: 7,
};

export interface BenchmarkPoint {
  series: SeriesId;
  n_samples: number;
  median_ms: number;
  points_per_sec: number;
}

export interface BenchmarkSummary {
  // The largest accelerated speedup (cpu_ms / accelerated_ms) across every size
  // both lanes measured — the headline "Nx faster" figure.
  peak_speedup: number;
  peak_speedup_n: number;
  // Accelerated throughput at the largest size both lanes reached.
  throughput_pts_per_sec: number;
}

// Peak speedup over the sizes both lanes measured, plus accelerated throughput at
// the largest such size. Pure so it can be unit-tested without workers.
export function summarize(
  cpu_ms: Map<number, number>,
  accel: Map<number, { ms: number; pts_per_sec: number }>,
): BenchmarkSummary {
  let peak_speedup = 0;
  let peak_speedup_n = 0;
  let throughput_pts_per_sec = 0;
  let largest_shared = 0;

  for (const [n, accel_point] of accel) {
    const cpu = cpu_ms.get(n);
    if (cpu === undefined) continue;
    const speedup = cpu / accel_point.ms;
    if (speedup > peak_speedup) {
      peak_speedup = speedup;
      peak_speedup_n = n;
    }
    if (n > largest_shared) {
      largest_shared = n;
      throughput_pts_per_sec = accel_point.pts_per_sec;
    }
  }

  return { peak_speedup, peak_speedup_n, throughput_pts_per_sec };
}
