---
id: TASK-55.5
title: Crossover n-slider with marked crossover point and flipping caption
status: Done
assignee: []
created_date: '2026-06-26'
updated_date: '2026-06-26'
labels:
  - demo
  - ux
dependencies:
  - task-55.4
parent_task_id: TASK-55
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->

Add the `n`-slider (200 to a hard cap of 5000) that re-runs the race live and flips the caption between "At small n, CPU wins — GPU transfer + dispatch cost more than the math" and "GPU pulls ahead." This is the most shareable, least cherry-pick-able interaction: making the small-`n` CPU win a first-class, visible part of the demo is what defends it against accusations of rigging.

Detect and mark the crossover `n` on the slider track — the point where the two wall-time curves intersect (fixed overhead: GPU dispatch + upload + single readback ≈ a few ms constant, CPU ≈ near zero; compute grows with `n²·d`). The 5000 cap is enforced because the dense `O(n²)` affinity matrix risks OOM/jank on low-end and mobile devices above it; large-`n` runs must not freeze the main thread (compute is in workers).

<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->

- [x] #1 Dragging the slider re-runs the race and updates bars/timers for the new `n`
- [x] #2 Below the crossover the CPU lane wins and the caption reflects it; above it the GPU wins
- [x] #3 The crossover `n` is computed and visibly marked on the slider track
- [x] #4 The slider cannot exceed `n=5000`; large-`n` runs do not freeze the main thread
- [x] #5 ESLint passes before commit

<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->

## High-level summary

The race carries an `n`-slider (200–5000) that re-runs the WebGPU-vs-CPU race live as `n` changes and flips a verdict caption between the small-`n` CPU win and the large-`n` GPU win. The caption is driven by the same measured `speedup` the headline tile reads, so it can never contradict the bars the visitor is watching. The crossover `n` — where the two wall-time curves intersect — is detected from the visitor's own measurements and marked on the slider track.

The defining decision is how the crossover is reported. The page's honesty contract forbids publishing a number it did not measure: the stopwatch only ever shows elapsed wall-clock, first-run shader-compile cost is excluded from the multiplier, and parity is a checksum rather than a claim. A model-fitted crossover — solving `a + b·n²` curves fitted to a handful of noisy single-race medians — would be an unfalsifiable prediction, exactly the kind of number the rest of the demo refuses to show. So the crossover is reported **only from an observed sign change** between two `n`'s the visitor has actually raced, linearly interpolated within that bracket. Until they have raced both a CPU-winning and a GPU-winning `n`, a directional hint points toward the unraced side rather than marking a guessed tick. This keeps the small-`n` CPU win a first-class, demonstrated part of the demo, which is what defends it against accusations of cherry-picking.

### What changed

- **`site/src/crossover.ts` (new)** — a pure, DOM-free estimator. `make_crossover_estimator()` accumulates `(n, cpu_ms, gpu_ms)` samples keyed by `n` (re-racing an `n` refines that point). `estimate()` returns one of four states: `unknown` (nothing raced), `bracketed` (an observed CPU→GPU sign change, with the interpolated crossover `n`), or `below_range` / `above_range` (every raced `n` falls on one side, so the crossover lies past the lowest / highest raced `n`). The ascending scan takes the lowest-`n` sign change, so run-to-run noise at large `n` cannot drag the mark around. Colocated `crossover.test.ts` covers the sign semantics, interpolation, the noise-stability rule, and the refine-on-re-race behaviour.
- **`site/src/race_ui.ts`** — `run()` now takes the active `n` and builds the config from `DEFAULT_RACE_CONFIG` (gamma stays `1/n_features` since `d` is fixed). A debounced single-flight scheduler (`schedule_race` → `drain`) coalesces drag events so only one race runs at a time and the latest `n` wins; all compute stays in workers, so the main thread never freezes even at the cap. The verdict caption participates in the busy/incomplete states so a re-run never shows a stale verdict, while the crossover mark — a cumulative estimate — persists across races. `make_race_ui` now owns the slider and run-button wiring directly.
- **`site/index.html` / `site/src/style.css`** — the slider (`min=200 max=5000 step=50`), its live readout, the crossover mark with its three visible states, and the flipping caption. The bracketed label anchors inward near the track ends so it never spills off-screen; the slider exposes `aria-valuetext` so screen readers announce the value with its unit.
- **`site/src/main.ts`** — reduced to constructing the controller, which wires its own listeners.

### Honesty and the `n=5000` cap

The cap is enforced on the input (`max=5000`) and re-clamped in `read_slider_n` as defence in depth — a tampered value can never push `n` past the dense `O(n²)` affinity matrix's memory ceiling. On a machine without WebGPU the "GPU" lane is honestly relabelled (e.g. WEBGL) by the existing harness, and the crossover story still holds: a GPU-class backend's fixed dispatch/upload/readback overhead dominates at small `n` and amortizes as the `O(n²·d)` compute grows. If WebGL never beats CPU within range on a given device, no bracket forms and the page shows a directional hint rather than asserting a win it cannot demonstrate.

### Review

Six independent reviewers (correctness ×2, contracts/integration, completeness-vs-spec, IA, adversarial cold-read) confirmed the scheduler is sound (single-flight holds, no unbounded recursion, no division by zero, consistent tie semantics between caption and estimator) and the variable-`n` data flow is correct. Three verified findings were fixed: a stale verdict caption on the failure / in-flight path, a bracketed label that could overflow near the track ends, and a screen-reader gap on the slider value. Lower-confidence and polish notes (a state-name rename, hint-copy phrasing, cosmetic thumb-inset tick alignment) were recorded but not actioned.

### Deferred

Lowering the cap specifically on detected mobile devices remains an open question — iOS 26 WebGPU is new and the tested-safe max `n` before OOM is unknown. The universal 5000 hard cap is shipped; a device-specific lower cap is a follow-up, not a requirement of this task.

<!-- SECTION:NOTES:END -->
