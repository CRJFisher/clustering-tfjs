// Crossover detection for the n-slider. The crossover is the n where the CPU and
// GPU wall-time curves intersect: below it the GPU's fixed dispatch/upload/readback
// overhead dominates and CPU wins; above it the O(n²·d) affinity compute dominates
// and the GPU wins.
//
// This page's honesty contract forbids publishing a number it did not measure (the
// stopwatch never shows a measured figure, first-run compile is excluded from the
// multiplier, parity is a checksum not a claim). A model-fitted crossover — solving
// a + b·n² curves fitted to a handful of noisy single-race medians — would be an
// unfalsifiable prediction, exactly the kind of number the rest of the demo refuses
// to show. So the crossover is reported ONLY from an observed sign change between
// two n's the visitor actually raced, linearly interpolated within that bracket.
// Until the visitor has raced both a CPU-winning and a GPU-winning n, there is no
// defensible point to mark — the estimator reports which direction to drag instead.

export const N_MIN = 200;
export const N_MAX = 5000;

export interface RaceSample {
  n: number;
  cpu_ms: number;
  gpu_ms: number;
}

// `bracketed` is the only state that places a tick. `below_range` / `above_range`
// mean every raced n fell on one side, so the crossover lies past the lowest /
// highest n raced so far — the UI shows a directional hint, never a guessed tick.
export type CrossoverState =
  | { kind: "unknown" }
  | { kind: "below_range" }
  | { kind: "above_range" }
  | { kind: "bracketed"; n: number };

export interface CrossoverEstimator {
  add_sample(sample: RaceSample): void;
  estimate(): CrossoverState;
}

// gpu_ms − cpu_ms: positive when the GPU is slower (CPU wins, small-n regime),
// negative when the GPU is faster (GPU wins, large-n regime). The crossover is
// where this delta passes through zero. Defined off the medians the workers
// reported, the same figures the headline tiles display.
function delta_gpu_minus_cpu(sample: RaceSample): number {
  return sample.gpu_ms - sample.cpu_ms;
}

export function make_crossover_estimator(): CrossoverEstimator {
  // Keyed by n so re-racing the same n refines that point rather than appending a
  // duplicate that would distort the ascending scan. Latest measurement wins.
  const by_n = new Map<number, RaceSample>();

  function estimate(): CrossoverState {
    if (by_n.size === 0) return { kind: "unknown" };

    const samples = [...by_n.values()].sort((a, b) => a.n - b.n);

    // Scan ascending for the first CPU-wins → GPU-wins transition. Taking the
    // lowest-n sign change keeps the mark stable: run-to-run noise at large n
    // can produce a spurious second flip, but the first crossing is the physical
    // one (overhead-dominated below, compute-dominated above).
    for (let i = 0; i < samples.length - 1; i++) {
      const lo = samples[i];
      const hi = samples[i + 1];
      const delta_lo = delta_gpu_minus_cpu(lo);
      const delta_hi = delta_gpu_minus_cpu(hi);
      if (delta_lo > 0 && delta_hi <= 0) {
        // Linear interpolation of the zero crossing between the two bracketing n's.
        const span = delta_lo - delta_hi;
        const n = lo.n + ((hi.n - lo.n) * delta_lo) / span;
        return { kind: "bracketed", n };
      }
    }

    // No sign change observed: every raced n is on one side. A positive delta at
    // the largest raced n means CPU is still winning there, so the crossover is at
    // a larger n (drag right); otherwise GPU already wins everywhere raced and the
    // crossover is at a smaller n (drag left).
    const largest = samples[samples.length - 1];
    return delta_gpu_minus_cpu(largest) > 0
      ? { kind: "above_range" }
      : { kind: "below_range" };
  }

  return {
    add_sample(sample: RaceSample): void {
      by_n.set(sample.n, sample);
    },
    estimate,
  };
}
