import * as tf from "@tensorflow/tfjs-core";
import { SpectralClustering } from "clustering-tfjs";

// The fairness protocol, backend-agnostic. Runs inside whichever worker
// initialized a backend; the timed region brackets exactly the real published
// RBF-affinity construction plus its single awaited readback. This is the
// headline workload: an O(n²·d) matmul/exp block where the GPU win is real and
// large, and — unlike fit/fit_predict — it performs no internal sync readback,
// so it is safe on the async-only WebGPU backend.

export interface HarnessConfig {
  // The float32 input, already uploaded to this worker's engine ONCE, outside
  // any timed region. Persists across every rep; the harness never disposes it.
  x: tf.Tensor2D;
  gamma: number;
  warmups: number;
  reps: number;
  on_phase?: (phase: "warmup" | "timed", rep: number) => void;
}

export interface HarnessResult {
  median_ms: number;
  min_ms: number;
  max_ms: number;
  first_run_ms: number;
  reps_ms: number[];
  tensors_baseline: number;
  result_checksum: number;
}

// One timed unit: dispatch the affinity compute, await the single readback, then
// dispose the returned matrix. compute_affinity_matrix is internally tidy-wrapped
// and returns one tensor, so disposing it here is the whole cleanup — numTensors
// returns to baseline. On WebGPU the dispatch is deferred, so the GPU's real
// execution cost lands inside `await affinity.data()`, keeping the number honest.
async function time_affinity(
  x: tf.Tensor2D,
  gamma: number,
): Promise<{ ms: number; data: Float32Array }> {
  const t0 = performance.now();
  const affinity = SpectralClustering.compute_affinity_matrix(x, {
    n_clusters: 2,
    affinity: "rbf",
    gamma,
  });
  const data = (await affinity.data()) as Float32Array;
  const t1 = performance.now();
  affinity.dispose();
  return { ms: t1 - t0, data };
}

// An order-sensitive sum of the affinity entries. CPU and GPU float32 paths
// should agree to several significant figures; a gross divergence signals one
// lane silently fell back or computed a different kernel.
function checksum(data: Float32Array): number {
  let sum = 0;
  for (let i = 0; i < data.length; i++) sum += data[i];
  return sum;
}

export async function run_affinity_bench(
  config: HarnessConfig,
): Promise<HarnessResult> {
  const { x, gamma, warmups, reps, on_phase } = config;

  // The protocol is only honest if its floors hold; enforce them here rather
  // than trusting every caller, so the harness cannot be silently weakened into
  // a compile-inclusive median or a single-best-run number.
  if (warmups < 2) {
    throw new Error(`Fairness protocol requires >= 2 warmups, got ${warmups}.`);
  }
  if (reps < 5) {
    throw new Error(`Fairness protocol requires >= 5 timed reps, got ${reps}.`);
  }

  // Warmups absorb WGSL shader compilation / wasm instantiation / lazy kernel
  // registration at this exact n,d shape. The first warmup is the cold,
  // compile-inclusive run; it is reported separately and never enters the median.
  let first_run_ms = NaN;
  for (let i = 0; i < warmups; i++) {
    on_phase?.("warmup", i);
    const { ms } = await time_affinity(x, gamma);
    if (i === 0) first_run_ms = ms;
  }

  // Baseline captured after warmups, with the persistent input already uploaded:
  // steady state should sit at exactly this count, and each rep must return to it.
  const tensors_baseline = tf.memory().numTensors;

  const reps_ms: number[] = [];
  let result_checksum = NaN;
  for (let i = 0; i < reps; i++) {
    on_phase?.("timed", i);
    const { ms, data } = await time_affinity(x, gamma);
    reps_ms.push(ms);
    if (i === 0) result_checksum = checksum(data);

    // Only genuine accumulation aborts the race: numTensors growing past the
    // baseline means a rep failed to release the affinity matrix. A count at or
    // below baseline is never a leak, so a benign lazy-backend transient cannot
    // false-trip and discard an otherwise-valid measurement.
    const live = tf.memory().numTensors;
    if (live > tensors_baseline) {
      throw new Error(
        `Tensor leak: numTensors grew to ${live} from baseline ${tensors_baseline} after rep ${i}`,
      );
    }
  }

  const sorted = [...reps_ms].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  const median_ms =
    sorted.length % 2 === 1
      ? sorted[mid]
      : (sorted[mid - 1] + sorted[mid]) / 2;

  return {
    median_ms,
    min_ms: sorted[0],
    max_ms: sorted[sorted.length - 1],
    first_run_ms,
    reps_ms,
    tensors_baseline,
    result_checksum,
  };
}
