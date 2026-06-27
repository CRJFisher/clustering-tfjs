---
id: TASK-55.8
title: Per-algorithm parameter sliders with plain-English captions
status: Done
assignee: []
created_date: '2026-06-26'
updated_date: '2026-06-27'
labels:
  - demo
  - ux
dependencies:
  - task-55.7
parent_task_id: TASK-55
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->

Add distill-style live sliders driving the grid: `n_clusters`; Spectral affinity/`gamma`; HDBSCAN `min_cluster_size`; Agglomerative `linkage`; SOM grid size — each with a one-line plain-English caption explaining its effect. This makes the page link-worthy as an educational explainer (earning durable inbound links and embeds), not just a benchmark.

Changing a slider re-clusters the affected cells live. Controls map to the real library parameter names so the copy-paste code panel (task-55.9) stays truthful.

<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->

- [ ] #1 Each listed algorithm exposes its key parameter(s) as a live control with a one-line caption
- [ ] #2 Adjusting a slider re-clusters and re-renders the relevant grid cells in real time
- [ ] #3 Controls map to the real library parameter names and produce correct clusterings
- [ ] #4 ESLint passes before commit

<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->

## High-level summary

A live control strip sits above the scikit-learn parity grid and re-clusters it in
real time. It exposes a global cluster count (driving K-Means, Spectral,
Agglomerative and SOM), Spectral affinity and RBF γ, HDBSCAN `min_cluster_size`,
Agglomerative linkage, and SOM grid size — each with a plain-English caption that
names the real library parameter, so the copy-paste code panel (task-55.9) stays
truthful.

The design turns on one decision: the grid curates _different_ params per
(algorithm, dataset) so every algorithm is shown at its best (Spectral uses
nearest-neighbors on the curved rows but RBF on blobs; Agglomerative uses single
linkage on the thin manifolds but Ward on blobs; the cluster count is 2 for the
two-shape rows and 3 elsewhere). A single global slider cannot express that, so
every control has an **Auto** state meaning "use the curated per-dataset value."
Auto reproduces grid_config's grid byte-for-byte, parity badges and all. The moment
a control leaves Auto it applies that parameter globally to its column(s); the
affected cells re-cluster and **drop their scikit-learn parity claim** — badge,
annotation, hover tooltip, and canvas aria-label all switch to a neutral "your
params / not checked vs sklearn" — because that override combination was never
verified against scikit-learn. Reset restores Auto and the verified grid. This
honours the page's standing honesty contract: it never asserts a parity it did not
measure.

## How the acceptance criteria are met

- **#1 — each algorithm exposes its key parameter(s) with a one-line caption:** the
  six controls above, built in `grid_controls_ui.ts` from a spec list; captions name
  the real parameters and explain the effect in plain English.
- **#2 — adjusting a control re-clusters the relevant cells in real time:** the grid
  worker is now long-lived. After the initial 25-cell sweep it stays up holding the
  warm tfjs backend and the uploaded datasets, and `run_grid` returns a controller
  whose `recluster(jobs)` re-fits cells off the main thread. Slider input is
  debounced (140 ms) and single-flighted, so a drag coalesces into one in-flight
  batch and only the cells whose effective params actually changed are re-fit.
- **#3 — controls map to real library parameter names and produce correct
  clusterings:** `grid_controls.ts` is a pure, unit-tested resolver from
  (curated params + overrides) → the exact discriminated-union `GridParams` the
  estimators accept, including correctly adding/dropping `gamma` vs `n_neighbors`
  when Spectral affinity switches.
- **#4 — ESLint passes:** clean, alongside `tsc --noEmit`, the production build, and
  59 colocated tests.

## Structure

- `grid_controls.ts` — pure logic: `ControlOverrides`, `resolve_params`,
  `is_overridden`/`params_equal` (the parity-drop predicate, defined off resolved-vs-
  curated params so an inert override — e.g. γ on a nearest-neighbors cell — keeps
  its badge), `resolve_all`, `NUMERIC_CONTROL_BOUNDS`, `clamp_numeric`. Colocated
  test covers all five algorithms, the Spectral affinity/γ interactions, and the
  curated-value-is-not-an-override cases.
- `grid_controls_ui.ts` — the control widgets; owns the live `ControlOverrides`.
- `grid_ui.ts` — orchestration: builds the grid, owns the worker controller and the
  single-flight re-cluster scheduler, and applies the parity-drop visuals. The
  parity decision is **snapshotted per dispatched job** so a control moving while a
  fit is in flight can never paint a verified badge over labels computed with other
  params.
- `grid_worker.ts` / `grid_protocol.ts` / `grid.ts` — the keep-alive worker, the
  `recluster`/`recluster_done` protocol, and the `GridController`.

## Review notes

A multi-lens review surfaced and fixed, before completion: the overridden cells'
hover tooltip still asserted sklearn parity (now swapped with the badge); an
async window where a control moving mid-fit could mislabel a result (fixed by
snapshotting the parity decision at dispatch); the reset button mislabelling Auto
as "scikit-learn defaults" (Auto is the curated grid, not library defaults); a
worker crash after init that could wedge the re-cluster latch (the controls now
disable and the latch clears when the backend reports non-live); plus dead code
(`CONTROL_ALGORITHMS`, `GridController.dispose`) and accessibility gaps (Auto
checkbox naming, form `aria-busy`, live canvas aria-labels).

<!-- SECTION:NOTES:END -->
