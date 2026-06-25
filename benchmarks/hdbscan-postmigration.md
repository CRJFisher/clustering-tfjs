# HDBSCAN Post-Migration Benchmark (task-54.10)

Float32 tensor front-half, JS float64 tail. Front-half (distance matrix → core distances
→ mutual reachability) runs on the TensorFlow.js backend in a fused `tf.tidy` with one
`Float32Array` readback at the MST boundary. Tail (MST, condensed tree, EoM) is plain JS.

## Timing: cpu backend (5 repeats, median)

| Config | n × d | Median (ms) | Min (ms) | Max (ms) | Tensor Δ |
|--------|-------|-------------|----------|----------|----------|
| small | 100×10 | 14.18 | 8.48 | 35.99 | 0 |
| medium | 1000×50 | 684.57 | 654.95 | 801.64 | 0 |
| hdbscan_n2000_d2 | 2000×2 | 2546.55 | 2204.89 | 2613.86 | 0 |
| hdbscan_n2000_d16 | 2000×16 | 2281.04 | 2262.64 | 2307.28 | 0 |
| hdbscan_n2000_d64 | 2000×64 | 2705.01 | 2682.76 | 2783.41 | 0 |
| hdbscan_n2000_d128 | 2000×128 | 3318.61 | 3273.48 | 3833.76 | 0 |
| hdbscan_n5000_d16 | 5000×16 | 15242.77 | 14590.54 | 18044.39 | 0 |
| hdbscan_n5000_d128 | 5000×128 | 28000.25 | 22696.25 | 31987.06 | 0 |

## Timing: tensorflow backend (5 repeats, median)

| Config | n × d | Median (ms) | Min (ms) | Max (ms) | Tensor Δ |
|--------|-------|-------------|----------|----------|----------|
| small | 100×10 | 1.66 | 1.59 | 61.12 | 0 |
| medium | 1000×50 | 36.94 | 31.17 | 61.65 | 0 |
| hdbscan_n2000_d2 | 2000×2 | 114.47 | 113.49 | 190.37 | 0 |
| hdbscan_n2000_d16 | 2000×16 | 166.51 | 119.50 | 212.65 | 0 |
| hdbscan_n2000_d64 | 2000×64 | 167.95 | 122.10 | 246.25 | 0 |
| hdbscan_n2000_d128 | 2000×128 | 132.06 | 111.94 | 170.12 | 0 |
| hdbscan_n5000_d16 | 5000×16 | 684.45 | 674.23 | 722.92 | 0 |
| hdbscan_n5000_d128 | 5000×128 | 702.62 | 684.60 | 1122.43 | 0 |

## Comparison: cpu vs tensorflow (in-session) + vs float64-JS baseline

The "vs baseline" column compares today's cpu (TF interpreter) run against the pure float64-JS
baseline from a separate session (`benchmarks/hdbscan-baseline.yaml`). The cpu column being
**slower** than the baseline is expected: the TF cpu backend runs tensor ops through the TF
interpreter, which has more overhead than the raw JS float64 loops the old pipeline used. The
`tensorflow` (tfjs-node) backend is the target and shows the real migration payoff. Treat the
in-session cpu vs tensorflow column as the authoritative speedup.

| Config | n × d | cpu (ms) | tensorflow (ms) | In-session speedup | vs baseline (cpu) |
|--------|-------|----------|-----------------|--------------------|--------------------|
| small | 100×10 | 14.18 | 1.66 | 8.55x faster | 2.91x slower |
| medium | 1000×50 | 684.57 | 36.94 | 18.53x faster | 3.02x slower |
| hdbscan_n2000_d2 | 2000×2 | 2546.55 | 114.47 | 22.25x faster | 2.72x slower |
| hdbscan_n2000_d16 | 2000×16 | 2281.04 | 166.51 | 13.70x faster | 2.45x slower |
| hdbscan_n2000_d64 | 2000×64 | 2705.01 | 167.95 | 16.11x faster | 2.50x slower |
| hdbscan_n2000_d128 | 2000×128 | 3318.61 | 132.06 | 25.13x faster | 2.81x slower |
| hdbscan_n5000_d16 | 5000×16 | 15242.77 | 684.45 | 22.27x faster | 2.43x slower |
| hdbscan_n5000_d128 | 5000×128 | 28000.25 | 702.62 | 39.85x faster | 3.15x slower |

## JS↔tfjs Crossover

JS↔tfjs crossover: tfjs first wins at **small** (n=100, d=10).

## Small-n JS Fallback Decision

**No small-n fallback added (YAGNI decision).**

At n=100 (small config): the `tensorflow` backend runs in 1.66 ms vs the float64-JS baseline
of 4.87 ms — a 2.9x speedup even at the smallest tested size. There is no regression. The
threshold for adding a fallback path (>1.5x tfjs overhead AND >5 ms absolute delta) is not
met at any config. A JS fallback would reintroduce the dual front-half code this task removed.

## Notes

- The dense O(n²) distance + mutual-reachability matrices remain the memory ceiling regardless
  of backend. This task improves constant factors and large-n / high-d throughput; the O(n²)
  memory wall is unchanged.
- The MST and condensed-tree tail remain the dominant cost at low dimensionality (low d means
  the O(n²·d) front-half is cheap relative to the O(n²) MST scan).
- Tensor count delta of 0 after fit confirms the tensor pipeline is leak-free (no dispose
  regression from task-54.8).
