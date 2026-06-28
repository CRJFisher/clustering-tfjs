---
id: TASK-54.9
title: Validate HDBSCAN parity and equivalence under the float32 front-half
status: Done
assignee: []
created_date: '2026-06-24'
updated_date: '2026-06-25'
labels:
  - hdbscan
  - testing
dependencies:
  - task-54.6
  - task-54.7
  - task-54.8
parent_task_id: TASK-54
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->

Confirm the migrated pipeline is correct end-to-end: labels match scikit-learn (and the 54.1 oracle) exactly up to permutation with consistent `-1` noise, and probabilities stay within the float32 bounds set in 54.2. Finalise the tie-free probability tolerance and the tie-bound MAE/agreement bounds from the _actual_ float32 outputs (54.2 set the structure and a provisional bound; this subtask tightens it to observed drift plus slack).

<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->

- [x] #1 Labels from the migrated pipeline match the 54.1 oracle exactly (up to cluster-id permutation, consistent noise) on every fixture, including the degenerate all-noise and single-blob cases
- [x] #2 Probabilities are within the 54.2 tolerances; the tie-free bound is finalised from the maximum observed float32 drift plus documented slack, and the tie-bound MAE/agreement bounds are confirmed against the float32 outputs (precomputed-cosine fixture checked explicitly)
- [x] #3 The full suite (`hdbscan.test.ts`, `kdistance.test.ts`, `condensation_tree.test.ts`, `minimum_spanning_tree.test.ts`) passes
- [x] #4 ESLint passes with no new errors (fix, do not ignore) before commit

<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
## High-level summary

The migrated float32 HDBSCAN pipeline passes parity validation on all 33 fixtures with zero label regressions. The measurement script (`scripts/measure_float32_drift.ts`) ran every fixture through the live float32 pipeline and compared outputs against both the sklearn oracle (in the fixture JSON) and the float64-JS oracle (`__fixtures__/hdbscan/__oracle__/hdbscan_oracle.json`).

**Label parity (AC#1):** Labels match sklearn exactly (up to cluster-id permutation, consistent `-1` noise) on all 33 fixtures, including the degenerate all-noise and single-blob cases. No gram-matrix cancellation flip was observed.

**Probability bounds (AC#2):** The key finding from the measurement is that for all tie-bound fixtures the float32 pipeline matches the float64 oracle with MAE = 0.000 exactly. All divergence from sklearn on tie-bound fixtures is purely MST tie-breaking (Prim vs numpy's unstable argsort), not float32 arithmetic. For tie-free fixtures, where the MST is unique, any divergence from sklearn is pure float32 accumulation error.

Measured drift across all 33 fixtures:

- **Tie-free max per-point drift vs sklearn:** 9.897e-5 on `nested_mcs8_ms2_{eom,leaf}_eps0.8` (both fixtures tie). All other tie-free fixtures: 3.165e-5 or 0.000. 1.5× headroom rounds to `1.5e-4`.
- **Tie-bound max MAE vs sklearn:** 0.1497 on `blobs_cosine_precomputed_mcs5` (the watch item from 54.2). All other tie-bound fixtures: ≤ 0.077. 20% headroom → 0.18 (unchanged).
- **Tie-bound min agreement vs sklearn:** 0.975 on `circles_mcs5_msdef_leaf_eps0.0` and `moons_mcs5_msdef_leaf_eps0.0`. Floor set at 0.94 (unchanged).

**Constant changes in `src/clustering/hdbscan.test.ts`:**

- `TIE_FREE_PROB_ATOL`: 1e-3 → **1.5e-4** (tightened from the provisional bound to the measured drift × 1.5)
- `TIE_BOUND_MAE_MAX`: 0.18 → **0.18** (unchanged; comments updated with measured 0.1497)
- `TIE_BOUND_AGREEMENT_MIN`: 0.94 → **0.94** (unchanged; comments updated with measured 0.975)
- `PROB_UPPER_BOUND`: 1 + 1e-6 → **1 + 1e-6** (unchanged; correct for float32 ULP headroom)

**AC#3 — full suite:** 159 tests pass (hdbscan.test.ts + kdistance.test.ts + condensation_tree.test.ts + minimum_spanning_tree.test.ts).

**AC#4 — ESLint:** passes with no new errors.

The measurement script (`scripts/measure_float32_drift.ts`) is kept alongside `scripts/hdbscan-oracle.ts` as a permanent diagnostic tool.

**Original implementation notes:**

Use `test_support/label_agreement.ts` (`alignment_agreement`, `labels_equivalent_with_noise`) as the oracle comparator. Treat a label mismatch versus the oracle as a real regression to fix (likely a gram-matrix cancellation flip — see 54.3's fallback), not a reason to weaken label parity. Only probability bounds may move, and only to track measured float32 drift.
<!-- SECTION:NOTES:END -->
