---
id: TASK-51
title: Rigorous SOM benchmarking against MiniSom reference
status: Done
assignee: []
created_date: '2026-06-07 18:27'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->

The SOM reference fixtures are generated from MiniSom but validated with very loose tolerances (weight average difference < 2.0, quantization-error relative error < 60%, label similarity > 0.5, with one fixture skipped). These loose bounds exist because our online mini-batch training diverges from MiniSom's batch training and the two implementations use different RNG and weight initialization, so the reference suite only catches gross regressions and shape errors, not subtle algorithmic correctness. Establish a numerically rigorous validation pipeline against MiniSom so the SOM implementation can be trusted at tight tolerances.

<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->

- [x] A deterministic batch training path produces SOM weights that match MiniSom's batch SOM to within a tight per-element tolerance (≤ 1e-3 on standardized data) for every reference config, given identical initial weights and decay schedule.
- [x] Reference fixtures pin the exact initial weight grid, and the test injects those initial weights into both the implementation and the reference run, so weight comparison is not confounded by RNG/init differences.
- [x] Learning-rate and radius decay schedules used to generate fixtures are identical to those used by the implementation under test, and the exact formulas are documented.
- [x] Quantization error and topographic error match the MiniSom reference within ≤ 1% relative error for every reference config.
- [x] BMU indices and resulting cluster labels match the reference exactly (or, where exactness is provably impossible, the documented near-exact bound is justified).
- [x] Reference coverage spans rectangular (with documented 4- vs 8-connectivity) and hexagonal topology, the gaussian and bubble neighborhood functions (plus mexican_hat where the implementation supports it), and a range of grid sizes and datasets.
- [x] No reference tests are skipped; the previously-skipped `blobs_10x10_gaussian_rectangular` case passes under the matched pipeline, or is removed with a documented, defensible reason.
- [x] The fixture-generation pipeline is reproducible and documented (venv setup, command, what each fixture field means, and the tolerance rationale).
- [x] The existing online-mode SOM property/self-consistency tests remain and are clearly separated from the numeric reference tests, so the production (online) training path is still exercised.
<!-- AC:END -->

## Implementation Plan

### 1. Characterize the divergence (spike)

Enumerate every difference between our SOM training and MiniSom that breaks numeric equivalence:

- Online mini-batch update vs MiniSom batch update (`train_batch`).
- Iteration semantics: MiniSom `num_iteration` (per-sample iterations) vs our `num_epochs` (full passes). The current generator stores `num_iteration` as `num_epochs`, which is not equivalent.
- Decay schedules for learning rate and radius (MiniSom default asymptotic decay `x / (1 + t/(T/2))`).
- Weight initialization and RNG (MiniSom seeded init ≠ our seeded init).
- Neighborhood normalization (Σ influence) and topology distance (rectangular 4/8 vs hexagonal axial).
  Write the findings into the task notes; they define exactly what must be matched.

### 2. Choose the equivalence strategy

Prefer adding a deterministic **batch reference training mode** to our SOM that replicates MiniSom's batch update exactly: per epoch, accumulate `Σ h·x` and `Σ h` over the whole dataset and set `w ← Σ(h·x) / Σ(h)` with the matched decay schedule. A batch mode is far easier to make numerically equivalent than reconciling two online sample orders, and is independently useful. (Alternative considered: port MiniSom's online order + RNG — rejected as brittle.)

### 3. Eliminate init as a variable

Add an `initial_weights` injection path to `SOM` (accept a `[grid_height][grid_width][n_features]` array). The generator exports the exact initial weight grid it seeds MiniSom with; the test feeds the identical grid to our SOM. This removes RNG/init divergence entirely.

### 4. Match decay schedules

Make the implementation's learning-rate and radius decay identical to MiniSom's for the reference mode (document the exact closed form). Verify by comparing the per-epoch lr/radius sequences.

### 5. Regenerate fixtures with the matched pipeline

Update `tools/sklearn_fixtures/generate_som.py` to: export `initial_weights`; train MiniSom in batch mode with the matched decay and correct epoch↔iteration mapping; expand scenarios (add mexican_hat where supported, more grid sizes). Each fixture stores `X`, `params`, `initial_weights`, final `weights`, `bmus`, `labels`, `quantization_error`, `topographic_error`, `u_matrix`.

### 6. Tighten the reference tests

Rewrite `src/clustering/som_reference.test.ts` to inject `initial_weights`, run batch reference-mode training, and assert: weights ≤ 1e-3 per element, QE/TE ≤ 1% relative, exact BMU/label match. Remove the `it.skip`.

### 7. Keep the production path covered

Leave the online-mode invariant tests in `som.test.ts` / `som_hexagonal.test.ts` intact and clearly labelled, so the default (online) training path is still validated for its own properties.

### 8. Document

Add a SOM benchmarking guide (under `docs/` or `backlog/docs/`) covering regeneration steps, the batch-equivalence guarantee, and tolerance rationale. Note that fixture regeneration requires the `tools/sklearn_fixtures/.venv` (MiniSom) and is not run in CI; the reference Jest suite is.

### Key files

- `src/clustering/som.ts` — batch reference training mode, `initial_weights` injection.
- `src/clustering/som_neighborhood.ts` — decay/neighborhood/topology parity.
- `tools/sklearn_fixtures/generate_som.py` — export initial weights, matched batch training + decay, expanded scenarios.
- `__fixtures__/som/*.json` — regenerated fixtures.
- `src/clustering/som_reference.test.ts` — tight tolerances, no skips.
- `docs/` or `backlog/docs/` — SOM benchmarking guide.

### Relationship to prior work

Builds on task-37 (SOM algorithmic correctness fixes) and task-33 (SOM two-phase clustering path). This task does not change the default online training behavior; it adds a verifiable reference mode and rebuilds the fixtures so the existing loose tolerances can be replaced with rigorous ones.

## Implementation Notes

### High-level summary

The SOM has two training paths, validated independently. The production path
(`SOM.fit`) is online mini-batch training, unchanged by this task. The new
reference path (`train_minisom_reference` in `src/clustering/som_reference_training.ts`)
is a tensor-free transcription of MiniSom's `train_batch` used only for numeric
validation. Because `train_batch` is deterministic once the initial weights are
fixed, injecting identical initial weights into MiniSom (at fixture-generation
time) and into the reference trainer (at test time) makes the two match to
floating-point precision. This replaces the previous loose tolerances (per-element
weight average difference < 2.0, QE relative error < 60%, one skipped fixture)
with per-element parity at 1e-9 for weights/QE/U-matrix, exact BMU/labels, and
topographic error within the acceptance bound — across 32 fixtures spanning four
datasets, square and non-square grids, both topologies, and all three
neighborhood functions, with nothing skipped.

### Key correction to the original plan

The task's Implementation Plan recommended a "batch-map" mode
(`w <- sum(h*x) / sum(h)`), believing MiniSom's `train_batch` is a batch SOM. It is
not: `train_batch` is deterministic **online-sequential** training (sample
`data[t mod n]`, single-sample update `w += eta(t)*g(t)*(x - w)` over the whole
grid, asymptotic per-iteration decay `p0 / (1 + t/(T/2))`). A batch-map mode
would not reproduce MiniSom and so could not validate against it. The reference
trainer therefore replicates the real `train_batch`, which both matches MiniSom
to ~1e-12 (far exceeding the <=1e-3 criterion) and is robust because the only
stochastic input — the initial weights — is injected. The plan's stated reason
for rejecting this ("brittle RNG order") does not apply: `train_batch`'s order is
deterministic.

### What was built

- **Reference trainer** (`src/clustering/som_reference_training.ts`): plain-array
  (not tensor) replication of MiniSom — asymptotic decay; separable gaussian,
  open-box bubble (integer indices, strict inequality), and Ricker mexican_hat
  neighborhoods; hexagonal offset coordinates (`(grid_height - 1 - row)` parity,
  `Y_HEX_CONV_FACTOR`); MiniSom's `[width][height]` weight axes transposed at the
  boundary to the library's `[height][width]`; and matching quantization error,
  topographic error (rectangular `> 1.42`, hexagonal numpy-`isclose` on hex
  coordinates), and `distance_map` U-matrix.
- **`initial_weights` production feature** (`SOMParams`, `som.ts`): an optional
  `[grid_height][grid_width][n_features]` grid honored by `fit` and `partial_fit`
  for reproducible training and warm-start, with shape and feature-dimension
  validation. It is excluded from persisted snapshots (the trained weights are
  saved separately).
- **Fixture generator** (`tools/sklearn_fixtures/generate_som.py`): injects and
  exports deterministic initial weights, uses MiniSom's native metrics, transposes
  axes (correct for non-square grids), uses snake_case schema with the unambiguous
  `num_iteration` field, expands coverage (mexican_hat + 8x4/4x8 grids), fixes the
  BMU/label convention, and pins its dependencies. 32 fixtures regenerated.
- **Reference suite** (`som_reference.test.ts`): rewritten to inject
  `initial_weights`, train once per fixture, and assert per-element weights/QE/
  U-matrix at 1e-9, exact BMU/labels, and TE within 1% relative (with an
  exact-zero guard); no skips.
- **Suite separation**: the online-mode property suites (`som.test.ts`,
  `som_hexagonal.test.ts`) are kept and relabeled with headers cross-referencing
  the reference suite.
- **Docs**: `docs/som-benchmarking.md` (two paths, formulas, connectivity, fixture
  schema, tolerance rationale, regeneration steps, CI policy), linked from
  `docs/debugging-guide.md`; `initial_weights` added to `docs/API.md`.

### Tolerance rationale

Weights, QE, and the U-matrix are continuous functions of the matched weights and
match to ~1e-12; they are asserted at 1e-9. BMU indices and labels are integer
argmins over matched weights and are asserted exactly. Topographic error is a
discrete metric whose second-nearest-neuron selection is discontinuous at distance
degeneracies, so a sub-1e-9 weight difference can flip one sample's classification;
it is held to the acceptance criterion's <=1% relative bound (observed: 31/32
fixtures at 0% deviation, one at 0.24%).

### Modified or added files

- Added: `src/clustering/som_reference_training.ts`, `docs/som-benchmarking.md`,
  16 new `__fixtures__/som/*.json`, the committed comprehension companion.
- Modified: `src/clustering/types.ts`, `src/clustering/som.ts`,
  `src/clustering/som_reference.test.ts`, `src/clustering/som.test.ts`,
  `src/clustering/som_hexagonal.test.ts`, `tools/sklearn_fixtures/generate_som.py`,
  `tools/sklearn_fixtures/requirements.txt`, `docs/debugging-guide.md`,
  `docs/API.md`, 16 regenerated `__fixtures__/som/*.json`.
