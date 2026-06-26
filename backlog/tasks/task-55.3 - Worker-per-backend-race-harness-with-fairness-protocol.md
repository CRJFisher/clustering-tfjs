---
id: TASK-55.3
title: Build one-worker-per-backend race harness with honest fairness protocol
status: To Do
assignee: []
created_date: '2026-06-26'
updated_date: '2026-06-26'
labels:
  - demo
  - webgpu
  - benchmark
dependencies:
  - task-55.1
  - task-55.2
parent_task_id: TASK-55
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->

Implement `race_worker.ts`, spawned via `new Worker(new URL('./race_worker.ts', import.meta.url), { type: 'module' })`, **one worker per backend**. Each worker imports `tfjs-core` + exactly one backend + the `clustering-tfjs` ESM classes and runs its own `Clustering.init` on its own tfjs engine. This worker-per-backend design is required, not preferred: tfjs has a single global engine per JS context, so WebGPU and CPU cannot run concurrently on one thread.

Implement the shared bench harness, enforced identically in every worker and surfaced in the methodology note (task-55.6):

1. Generate ONE float32 input tensor; upload identical data to every lane (never compare a float64 JS path against float32 GPU — the "speedup" would be partly a precision artifact).
2. 2–3 discarded warmup runs per backend (absorb WGSL shader compile, wasm instantiation, lazy kernel registration).
3. Time the full real **Spectral RBF affinity** workload **including the awaited readback** (`await tensor.data()`) — timing only JS dispatch produces a fake ~100× and reads as dishonest.
4. Exactly ONE readback boundary per run, identical across lanes.
5. ≥5 timed reps; report **median** plus min/max, never a single best run.
6. `tidy`/dispose between runs; assert `tf.memory().numTensors` returns to baseline so GC/leaks don't skew results.
7. Capture the first-run (shader-compile-inclusive) number separately so the UI can disclose it without putting it in the headline.

Default workload: Spectral RBF affinity (`O(n²·d)`), make_blobs `n=2000`, `d=32` (computed in 32 dims, rendered in 2). **Not** KMeans — its Lloyd loop reads back every iteration and yields only ~1.2×. Must call the real published `SpectralClustering`, never `benchmarks/browser_backend.ts` (simulated fake kmeans).

<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->

- [ ] #1 WebGPU and CPU each run in their own worker concurrently with independent tfjs engines
- [ ] #2 The timed region brackets the full `fit`/affinity call including `await tensor.data()` (not just JS dispatch)
- [ ] #3 Exactly one readback boundary per run, identical across lanes; median of ≥5 reps reported with min/max
- [ ] #4 `tf.memory().numTensors` returns to baseline between runs (asserted)
- [ ] #5 The harness invokes the published `clustering-tfjs` `SpectralClustering`, not the simulated `browser_backend.ts` kmeans
- [ ] #6 First-run shader-compile time is captured separately from the steady-state median
- [ ] #7 ESLint passes before commit

<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->

Reference anchor points the live numbers should approximate: Spectral affinity ~1410ms→67ms at 1000×50 (~21×); ~140,738ms→5,392ms at 10000×100 (~26×) on the native backend. The slider (task-55.5) hard-caps `n` at 5000 — above that the dense `O(n²)` matrix risks OOM/jank on low-end and mobile devices.

<!-- SECTION:NOTES:END -->
