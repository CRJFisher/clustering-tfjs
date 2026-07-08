---
id: TASK-55.3
title: Build one-worker-per-backend race harness with honest fairness protocol
status: Done
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

- [x] #1 WebGPU and CPU each run in their own worker concurrently with independent tfjs engines
- [x] #2 The timed region brackets the full `fit`/affinity call including `await tensor.data()` (not just JS dispatch)
- [x] #3 Exactly one readback boundary per run, identical across lanes; median of ≥5 reps reported with min/max
- [x] #4 `tf.memory().numTensors` returns to baseline between runs (asserted)
- [x] #5 The harness invokes the published `clustering-tfjs` `SpectralClustering`, not the simulated `browser_backend.ts` kmeans
- [x] #6 First-run shader-compile time is captured separately from the steady-state median
- [x] #7 ESLint passes before commit

<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->

## High-level summary

The race harness runs one Web Worker per backend. Because TensorFlow.js keeps a single global engine per JS realm, WebGPU and CPU can only race concurrently in separate workers, each with its own engine. Each worker imports `clustering-tfjs` (for `Clustering.init` and `SpectralClustering`) and `@tensorflow/tfjs-core` directly (to upload the input tensor, check `tf.memory()`, and dispose); Vite dedupes `tfjs-core` so both resolve to the one engine `Clustering.init` configures.

The timed workload is the published `SpectralClustering.compute_affinity_matrix(X, { affinity: 'rbf', gamma })` followed by exactly one `await affinity.data()`. This is the real library's RBF-affinity construction — the `O(n²·d)` matmul/exp block where the GPU win is large — and the only spectral step that performs no internal synchronous readback, so it is safe on the async-only WebGPU backend (`fit`/`fit_predict` are not). The simulated `benchmarks/browser_backend.ts` kmeans is never touched.

The fairness protocol lives in `bench_harness.ts` and is self-enforcing (it rejects fewer than 2 warmups or 5 timed reps): one float32 input uploaded once outside every timed region, 2–3 discarded warmups that absorb shader compile, the cold first run captured separately and kept out of the median, ≥5 timed reps reporting median plus min/max, the affinity matrix disposed each rep, and a per-rep assertion that `numTensors` never grows past baseline. Each lane reports the backend it actually ran, so a WebGPU→WebGL fallback is surfaced honestly and the headline names the real backend, never the requested one.

The screenshot-worthy dual-panel race UI (live timers, racing bars, speedup tiles) is task-55.4; this task ships a minimal button-and-text driver that exercises the harness end to end.

### What changed

- `site/src/race_protocol.ts` — the main↔worker message contract (`RaceRequest` / `RaceProgress` / `RaceResult` / `RaceError`), with `actual_backend` separate from the requested lane.
- `site/src/bench_harness.ts` — the backend-agnostic fairness protocol: warmups, median/min/max, separate first-run, per-rep dispose, and the `numTensors`-growth leak guard.
- `site/src/race_worker.ts` — one worker per backend: `init_lane` initializes the requested backend (WebGPU degrades to WebGL on missing `navigator.gpu` or init failure) and reports `tf.getBackend()`; the input tensor is uploaded once and disposed in a `finally`.
- `site/src/make_blobs_js.ts` — a dependency-free seeded (mulberry32 + Box–Muller) blob generator so the main thread never initializes tfjs and both lanes get byte-identical float32 input.
- `site/src/race.ts` — the main-thread orchestrator: generates one dataset, spawns both lanes via `Promise.allSettled` (so one lane's failure never strands the sibling worker), enforces a per-lane timeout, and reports the GPU lane's actual backend in the outcome.
- `site/src/main.ts` / `index.html` / `style.css` — the minimal run-the-race driver.

### Library consumption and CI

The site depends on the library through a `clustering-tfjs: file:..` link rather than the published npm package, so the demo always exercises the unreleased WebGPU lane from this branch. `deploy-site.yml` therefore builds the library before the site (`npm ci --ignore-scripts` skips tfjs-node's native binary and Puppeteer's Chromium — irrelevant to a browser build and a deploy-failure risk — and also skips the `prepare` build so the library builds exactly once). All four browser backends (cpu/webgl/webgpu/wasm) are installed pinned to 4.22.0 because the library loader statically references each via dynamic import. The library's Node/RN-only TensorFlow.js packages are externalized in both the main and worker Rollup configs so the browser bundle builds. Site sources are wired into ESLint via a self-contained `site/eslint.config.mjs` (the project's snake_case rules) and a `Lint site` step runs in CI on every PR.

### Review follow-ups folded in

A multi-agent review hardened the harness: the leak assertion fails only on genuine growth (not a benign lazy-backend transient that would abort a valid race); the warmup/rep floors are enforced inside the harness; a per-lane timeout plus `Promise.allSettled` guarantee no worker leaks when a lane hangs or fails; and the headline names the GPU lane's actual backend so a WebGL fallback can never be presented as a WebGPU win.

Reference anchor points the live numbers should approximate: Spectral affinity ~1410ms→67ms at 1000×50 (~21×); ~140,738ms→5,392ms at 10000×100 (~26×) on the native backend. The slider (task-55.5) hard-caps `n` at 5000 — above that the dense `O(n²)` matrix risks OOM/jank on low-end and mobile devices.

Task-55.2 left `site/` outside the repo's ESLint scope (its only TypeScript was a one-line CSS import, so linting it then was YAGNI). This sub-task adds the first real site logic, so wire `site/` into ESLint here with the project's Python-style snake_case rules — either extend the root flat config with a `site/src/**/*.ts` block or add a site-local config — and add an `npm run lint` step to the `deploy-site.yml` build job so the convention is enforced in CI (satisfies this task's AC #7 for the site sources).

<!-- SECTION:NOTES:END -->
