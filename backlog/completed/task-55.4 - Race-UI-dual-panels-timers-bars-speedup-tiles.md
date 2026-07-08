---
id: TASK-55.4
title: 'Race UI: dual scatter panels, live timers, racing bars, speedup tiles'
status: Done
assignee: []
created_date: '2026-06-26'
updated_date: '2026-06-26'
labels:
  - demo
  - ux
dependencies:
  - task-55.3
parent_task_id: TASK-55
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->

Build the hero race UI — the screenshot-worthy core and the launch's minimum viable asset. Two side-by-side panels (left = CPU, right = WebGPU/GPU), each with a lightweight 2D-canvas scatter of the same seeded dataset and, below it, a large live wall-clock timer and a horizontal bar that races left-to-right as the run completes.

Centered headline tiles: (1) ms per backend, (2) the "GPU is N.Nx faster" multiplier (the tweet line), (3) points/sec throughput, (4) a steady-state vs first-run toggle. The first-run (shader-compile) number appears only behind the toggle, never in the headline multiplier.

Display a "same result" indicator backed by an actual cross-backend label-equality check. All scatter rendering stays outside the timed window (no chart redraw inside measured runs) so the page never competes with the measured compute.

The top fold is designed to double as a 1200×630 og:image: fixed dark high-contrast theme, large legible timers/counters, high-contrast cluster colors.

<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->

- [x] #1 Both panels render the same seeded dataset and update timers/bars live from worker results
- [x] #2 The speedup multiplier tile shows the median-based "N.Nx faster" and matches the harness numbers
- [x] #3 First-run (shader-compile) time is shown only behind the toggle, never in the headline multiplier
- [x] #4 Scatter rendering occurs outside the timed region (no chart redraw inside measured runs)
- [x] #5 The "same result" indicator reflects an actual cross-backend result-equality check on the timed affinity output (matching `result_checksum` within relative tolerance) — see Implementation Notes for why this is the honest realization, not a label check
- [x] #6 ESLint passes before commit

<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->

## High-level summary

The hero race fold is the screenshot-worthy centerpiece of the demo: two side-by-side panels (CPU on the left, the GPU lane on the right) race the same seeded dataset, each showing a 2-D scatter of that data, a large live wall-clock timer, and a horizontal bar that fills as the run progresses. Below the panels sit four headline tiles — per-backend median, the "N.Nx faster" multiplier (the tweet line), throughput, and a steady-state-vs-first-run toggle — plus a cross-backend "same result" indicator and the permanent "numbers come from YOUR hardware" footer. The fold is built to double as the 1200×630 og:image: fixed dark high-contrast theme, large legible tabular-numeric counters, high-contrast cluster colors.

The UI is a pure orchestrator over the task-55.3 worker harness. It adds no compute on the main thread; it translates the harness's `on_progress` / `on_lane_result` / `on_lane_error` callbacks into cheap DOM updates and renders the scatter exactly once, before timing, through a new `on_dataset` callback on `RaceCallbacks`.

**Honesty is enforced by construction, not convention:**

- The big live timer is a `requestAnimationFrame` stopwatch that reports main-thread **elapsed** wall-clock (captioned "elapsed"), which freezes and snaps to the harness's measured `median_ms` (captioned "median · N reps") the instant a lane's result lands. The stopwatch primitive never receives the median, so it structurally cannot present its own wall-clock as the measured figure.
- The multiplier reads only `outcome.speedup` (the harness's median-based `cpu.median_ms / gpu.median_ms`). The cold first-run time lives solely behind the toggle and is never read by the headline, so shader-compile cost cannot leak into the claim.
- A small-`n` CPU win renders as a "CPU is faster" multiplier rather than a sub-1 number that reads as broken — the crossover is first-class, per the design.
- The GPU panel relabels to its **actual** backend (`actual_backend`), so a WebGL fallback can never be displayed as a WebGPU win.
- The race never competes with the GPU for frames: the scatter draws once, off the timed path; only the cheap timer/bar/tile nodes update during a run.

### The 2-D scatter (judgment call A)

The dataset is 32-dimensional, so the scatter needs a projection. `project_2d.ts` computes a deterministic 2-D PCA (power iteration with deflation on the 32×32 covariance, float64 accumulators, a fixed start vector, a fixed iteration count, and sign-pinned eigenvectors). PCA — not two raw feature axes — because the blobs' centers sit at random positions and two arbitrary axes rarely separate them; it is also the standard, defensible reduction for a technical audience and never bakes the ground-truth centers into the axes. Determinism is a hard requirement for a future shareable permalink, and `project_2d.test.ts` pins reproducibility, centering, variance ordering, and blob separation. The projection is computed once and rendered into both panels (`scatter_canvas.ts`, devicePixelRatio-aware, one-shot, no animation loop), so both panels show the identical seeded data.

### The "same result" indicator (judgment call B, AC#5)

AC#5 originally said "label-equality check," but the timed workload is Spectral **RBF affinity construction** (task-55.3's `bench_harness.ts`) — it produces an affinity matrix, never cluster labels. A literal label check would require `fit_predict`, whose internal **synchronous readback is unsafe on the async-only WebGPU backend** (the exact reason the harness times affinity construction and not `fit_predict`). The honest realization is therefore an equality check on the actual timed output: both lanes already return a `result_checksum` (an order-sensitive sum of the affinity matrix), and the indicator compares the two with a symmetric relative tolerance (`1e-3`, never dividing by a single lane's value). The match copy is deliberately "both backends agree on the RBF affinity matrix (matching checksum)" — not "identical matrix," since a scalar checksum proves the same kernel ran, not bit-for-bit equality. AC#5 here and the parent task-55 AC#2 were reconciled from "label-equality" to this honest result-equality wording.

### Module layout

- `project_2d.ts` (+ `project_2d.test.ts`) — deterministic 2-D PCA.
- `scatter_canvas.ts` — one-shot devicePixelRatio-aware point renderer.
- `stopwatch.ts` — the rAF elapsed-time primitive (elapsed only, never the median).
- `race_ui.ts` — the controller: a per-lane view factory, the headline render, the parity check, and the toggle wiring all live here.
- `main.ts` — reduced to a thin bootstrap.
- `race.ts` — gains the `on_dataset` callback (no behavior change to the existing race).

The site build's `tsc --noEmit` now typechecks the colocated test too (`@types/jest` added as a site devDep, with a `/// <reference types="jest" />` so the globals resolve under `bundler` module resolution); the test runs under the repo-root Jest.

### Review outcome

A six-lens opus review (correctness ×2, contracts, completeness-vs-spec, IA, adversarial cold-read) returned no blockers or majors against the code. Verified findings applied: softened the parity match copy away from "identical matrix"; gave an incomplete race (a crashed/timed-out lane) a distinct neutral "error" state instead of borrowing the divergence warning; added `aria-live="polite"` to the parity line so the result summary is announced once. Reconciling the AC wording (above) was the one major-severity finding, addressed in both docs. Noted-but-not-actioned (tracked, below the gate or deferred): throttling the rAF timer repaint under `prefers-reduced-motion`, fuller assistive-tech announcement of the medians, throughput-tile width on very narrow viewports, reading the cluster palette from CSS tokens at render time, and the cold fold's og:image-readiness (a task-55.10 concern).

<!-- SECTION:NOTES:END -->
