---
id: TASK-51
title: Rigorous SOM benchmarking against MiniSom reference
status: To Do
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

- [ ] A deterministic batch training path produces SOM weights that match MiniSom's batch SOM to within a tight per-element tolerance (≤ 1e-3 on standardized data) for every reference config, given identical initial weights and decay schedule.
- [ ] Reference fixtures pin the exact initial weight grid, and the test injects those initial weights into both the implementation and the reference run, so weight comparison is not confounded by RNG/init differences.
- [ ] Learning-rate and radius decay schedules used to generate fixtures are identical to those used by the implementation under test, and the exact formulas are documented.
- [ ] Quantization error and topographic error match the MiniSom reference within ≤ 1% relative error for every reference config.
- [ ] BMU indices and resulting cluster labels match the reference exactly (or, where exactness is provably impossible, the documented near-exact bound is justified).
- [ ] Reference coverage spans rectangular (with documented 4- vs 8-connectivity) and hexagonal topology, the gaussian and bubble neighborhood functions (plus mexican_hat where the implementation supports it), and a range of grid sizes and datasets.
- [ ] No reference tests are skipped; the previously-skipped `blobs_10x10_gaussian_rectangular` case passes under the matched pipeline, or is removed with a documented, defensible reason.
- [ ] The fixture-generation pipeline is reproducible and documented (venv setup, command, what each fixture field means, and the tolerance rationale).
- [ ] The existing online-mode SOM property/self-consistency tests remain and are clearly separated from the numeric reference tests, so the production (online) training path is still exercised.
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
