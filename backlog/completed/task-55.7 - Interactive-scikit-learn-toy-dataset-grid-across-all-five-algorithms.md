---
id: TASK-55.7
title: Interactive scikit-learn toy-dataset grid across all five algorithms
status: Done
assignee: []
created_date: '2026-06-26'
updated_date: '2026-06-27 09:30'
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
Recreate the iconic scikit-learn "Comparing clustering algorithms on toy datasets" image — but live and interactive. Rows = moons / circles / blobs / anisotropic / no-structure; columns = K-Means, Spectral, Agglomerative, HDBSCAN, SOM; each cell a live canvas scatter colored by cluster, computed via the real library in a worker. This section sells trust/parity: it signals "this is the sklearn clustering you already know, in your browser."

Compute runs off the main thread and re-renders without freezing the page. Curate datasets/params where float32 parity with sklearn holds; where it does not (the known HDBSCAN/Spectral float32 probability drift), annotate the expected difference rather than hiding it — the parity claim must not be undercut. KMeans may show animated convergence here (it is animation eye-candy, not a speedup claim).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 All five algorithms produce live cluster colorings across all five dataset rows
- [x] #2 Compute runs off the main thread and re-renders without freezing the page
- [x] #3 Datasets/params are curated so float32 labels match the sklearn-parity story, or differences are annotated
- [x] #4 The grid is visually legible with high-contrast (colorblind-safe) cluster colors at the hero framing
- [x] #5 ESLint passes before commit
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
## High-level summary

The demo now carries the section that sells trust: a live recreation of
scikit-learn's iconic "comparing clustering algorithms on toy datasets" figure.
Five dataset shapes (two moons, concentric circles, blobs, anisotropic, no
structure) form the rows; the library's five algorithms (K-Means, Spectral,
Agglomerative, HDBSCAN, SOM) form the columns; each of the 25 cells is a small
canvas scatter coloured by the labels the real library computes. The message is
"this is the scikit-learn clustering you already know, running in your browser" —
so the grid is about parity, not speed, and it reproduces scikit-learn faithfully
including the shapes where an algorithm is *meant* to fail.

The approach mirrors the established race fold (`*_protocol` / `*_worker` /
orchestrator / `*_ui`) and adds a single config module as the source of truth.
One worker initializes one fit-predict-safe backend (`webgl → wasm → cpu`, never
`webgpu` — synchronous tensor readback inside `fit_predict` is unsupported there)
and fits all 25 cells sequentially, streaming each cell's labels the instant they
land. A single sequential worker — not a pool — is the cheapest correct choice:
the toy data is tiny and the tfjs engine stays warm across cells. All compute is
off the main thread, so the page never freezes, and each cell renders exactly
once when its labels arrive.

Parity is told honestly. Each cell carries one of four tiers: cells whose float32
labels match scikit-learn's exactly are marked ✓; Spectral on the curved and
anisotropic rows and every HDBSCAN cell are marked ≈ ("cluster cores match,
boundary points can drift under float32") rather than tuned to hide the known
drift; SOM cells declare they have no scikit-learn counterpart; and the
no-structure row declares it has no true clusters (HDBSCAN correctly reports
all-noise there). The tier counts in the footnote are computed from config, so the
"9 of 25 match exactly" claim can never drift from what the badges show.

### Navigating the result

Start at `site/src/main.ts`, which mounts `make_grid_ui()` alongside the race UI.
`grid_config.ts` is the front door to the data: it declares the datasets,
algorithms, the 25-cell parameter table, and each cell's parity tier, and exports
`GRID_CELLS` (the render-order cross product) plus `cell_id_of`/`count_parity`.
`grid_ui.ts` builds the DOM from that config and wires the per-cell render;
`grid.ts` generates the datasets once and streams the worker's results; the
compute itself lives only in `grid_worker.ts`. The deterministic 2-D dataset
generators are in `make_toy_datasets.ts` (tested in its colocate).

### What changed

- **Toy-dataset generators (`site/src/make_toy_datasets.ts`).** Deterministic,
  seeded 2-D generators for moons / circles / isotropic blobs / anisotropic /
  no-structure, mirroring scikit-learn's `make_moons`/`make_circles`/aniso math.
  Every dataset is standardized to zero-mean/unit-variance so one curated set of
  per-cell params (gamma, n_neighbors, min_cluster_size) serves every row. The
  shared `mulberry32`/`next_gaussian` PRNG is now exported from `make_blobs_js.ts`
  so the whole site draws from one seeded stream. Colocated tests assert shape,
  determinism, standardization, label domains, and the anisotropic shear.
- **Grid fold (`grid_config.ts`, `grid_protocol.ts`, `grid_worker.ts`,
  `grid.ts`, `grid_ui.ts`).** Config-driven 5×5 grid; one streaming worker over a
  `webgl → wasm → cpu` chain; per-cell `fit_predict` (SOM uses the two-phase
  `fit` + `cluster(n)` path); honest per-cell parity badges and a config-computed
  footnote. A 120 s watchdog plus a terminal sweep in `on_done` mark any
  unreported cell failed and make the status line reflect what actually finished,
  so a timeout or worker crash never leaves the grid frozen on "Computing N / 25".
- **Noise rendering (`site/src/scatter_canvas.ts`).** `render_scatter` now draws
  label `< 0` (HDBSCAN noise) in a muted gray under the cluster colours, so the
  no-structure HDBSCAN cell reads as all-noise rather than blank.
- **Chained-op registration (`site/src/grid_worker.ts`).** A bare side-effect
  import of `@tensorflow/tfjs-core/.../register_all_chained_ops` pins the chained
  tensor methods (`x.square()`, `x.min()`, …) the library's algorithms call
  internally; the production bundler tree-shakes them out otherwise, since the
  worker never calls a chained op directly.

### Acceptance criteria

All five are met, confirmed by a headless-browser run of all 25 cells:

- **#1** — all five algorithms produce live colorings across all five rows (25/25
  cells computed, 0 errors on WebGL).
- **#2** — compute runs in the worker; the main thread never initializes tfjs and
  renders each cell once, so the page never freezes.
- **#3** — params are curated so float32 labels match the parity story, and the
  known HDBSCAN/Spectral float32 drift is annotated (≈ tier), not hidden. The
  task's open delivery question — the exact dataset+param combinations — is
  resolved in `grid_config.ts` and validated empirically.
- **#4** — cells use the high-contrast colorblind-safe palette; noise is gray.
- **#5** — ESLint passes.
<!-- SECTION:NOTES:END -->
