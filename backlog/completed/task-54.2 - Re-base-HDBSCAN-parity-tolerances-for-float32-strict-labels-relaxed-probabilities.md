---
id: TASK-54.2
title: >-
  Re-base HDBSCAN parity tolerances for float32 (strict labels, relaxed
  probabilities)
status: Done
assignee: []
created_date: '2026-06-24'
updated_date: '2026-06-25 09:56'
labels:
  - hdbscan
  - testing
dependencies:
  - task-54.1
parent_task_id: TASK-54
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The parity suite in `src/clustering/hdbscan.test.ts` asserts two tiers keyed off each fixture's `tie_free` flag: tie-free fixtures require exact label equivalence plus per-point probabilities to `toBeCloseTo(..., 6)` (~5e-7); tie-bound fixtures require exact cluster count, `alignment_agreement >= 0.95`, and probability MAE `<= 0.16`. The strict `1e-6` tier has zero float32 headroom and is the binding blocker for the front-half migration.

Re-base the tolerances to reflect the decided parity stance: **labels stay strict, probabilities relax**. Labels survive float32 (verified empirically on every fixture), so the label assertions are kept exactly as-is. The probability assertions are loosened to a float32-appropriate bound derived from measured drift plus deliberate slack, and the rationale is documented in the test header so the bound is meaningful rather than arbitrary.

This subtask lands the tolerance change before the tfjs stages so each stage can be validated against a suite that admits float32.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Tie-free fixtures keep exact `labels_equivalent_with_noise` assertions unchanged
- [x] #2 The tie-free per-point probability assertion is loosened from `toBeCloseTo(..., 6)` to a documented float32 bound (set from the measured front-half drift, ~1e-4, plus slack — e.g. ~1e-3)
- [x] #3 Tie-bound MAE and `alignment_agreement` bounds are re-measured under the float32 front-half and set from observed values plus slack (the precomputed-cosine fixture at 0.150 is the watch item)
- [x] #4 The test-header comment is rewritten to explain the float32 parity stance: why labels are exact but probabilities are bounded, replacing the float64 "exceeding the bound means re-measure not loosen" framing
- [x] #5 The suite passes against the float64 baseline with the new bounds (loosening a bound must never make a currently-passing assertion fail)
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
## High-level summary

The HDBSCAN parity suite is the gate every front-half tfjs stage must clear, and its strict probability tier had zero float32 headroom: it asserted per-point probabilities to ~5e-7, which the float32 tensor front-half cannot meet. This subtask re-bases the suite to the decided parity stance — **strict labels, bounded probabilities** — so the later tensor stages can be validated against a suite that admits float32 drift, while the suite still passes on today's float64 pipeline.

The four tolerance numbers become named `UPPER_SNAKE_CASE` module constants in `src/clustering/hdbscan.test.ts`, each documented with the observed value it derives from and the float32 headroom it carries. Labels keep their exact assertions, because float32 does not change which mutual-reachability edges the MST selects, so cluster membership is invariant. Probabilities, which depend on lambda ratios that float32 perturbs additively, move to bounded checks: the tie-free per-point assertion changes from `toBeCloseTo(..., 6)` to an absolute bound `TIE_FREE_PROB_ATOL = 1e-3` (~10x the ~1e-4 drift the migration probe measured); the tie-bound per-fixture MAE bound rises to `TIE_BOUND_MAE_MAX = 0.18` (the saturated-cosine fixture `blobs_cosine_precomputed_mcs5` is the watch item at 0.150, every other fixture ≤ 0.077); and the alignment floor relaxes to `TIE_BOUND_AGREEMENT_MIN = 0.94` (lowest observed 0.975). A universal `[0, 1]` probability guard widens to a float32-scale epsilon `PROB_UPPER_BOUND = 1 + 1e-6` at both the parity and API-surface call sites, since float32 can round a true 1.0 just above 1.

To navigate the result: the entire parity contract lives in `hdbscan.test.ts` — the four constants at the top, with the parity `describe` header explaining why labels are exact but probabilities bounded. The standalone-helper suites (`condensation_tree.test.ts`, `kdistance.test.ts`) stay float64-exact by design — they exercise the JS helpers directly and never touch the tensor front-half — so they are deliberately left unchanged.

What to know: every change is a pure loosening, so the float64 baseline still passes (58/58) and AC#5 holds by construction. The bounds are provisional — task-54.9 re-measures and tightens them against the migrated pipeline. The degenerate all-noise fixtures keep their exact `toEqual` checks because their probabilities are structural zeros (`Array.fill(0)` on unassigned points), not float32 arithmetic, so float32 cannot perturb them.

**Acceptance criteria addressed:**

- **AC#1** — The tie-free `labels_equivalent_with_noise(labels, fixture.labels)` assertion is untouched; labels stay exact up to cluster-id permutation with `-1` noise fixed.
- **AC#2** — The tie-free per-point probability check moves from `toBeCloseTo(..., 6)` (~5e-7) to `Math.abs(probs[i] - fixture.probabilities[i]) <= TIE_FREE_PROB_ATOL` with `TIE_FREE_PROB_ATOL = 1e-3`, documented at the constant from the ~1e-4 measured probe drift plus ~10x slack.
- **AC#3** — `TIE_BOUND_MAE_MAX = 0.18` (from observed 0.150 on `blobs_cosine_precomputed_mcs5` plus slack for float32 tie-reordering) and `TIE_BOUND_AGREEMENT_MIN = 0.94` (from observed min 0.975 plus slack) are named constants set from the float64 baseline; 54.9 re-measures under the real float32 outputs.
- **AC#4** — The parity `describe` header is rewritten to the float32 stance (labels exact because MST topology is invariant; probabilities bounded because lambda ratios drift additively), removing the float64 "exceeding the bound means re-measure, not loosen" framing.
- **AC#5** — The suite passes against the float64 baseline (58/58) with the new bounds; every bound change is a monotone loosening, so no currently-passing assertion can fail, and ESLint passes.

**Scope boundary (carried to 54.9):** `condensation_tree.test.ts` (`toBeCloseTo(..., 6)`, `>= 0.95`) and `kdistance.test.ts` (`toBeCloseTo(..., 10)`) keep their float64-exact tolerances. They validate the sequential JS-tail helpers directly and never run the tensor float32 front-half, so they must not be relaxed to match this suite.

Measured drift anchoring the bounds: tie-free probability overshoot was ≈9.9e-5 (`nested_mcs8_ms2_*_eps0.8`) and ≈3.2e-5 (`blobs_overlap_mcs10_ms2_*`); tie-bound float64 MAE peaks at 0.150 (`blobs_cosine_precomputed_mcs5`, every other fixture ≤ 0.077) and alignment bottoms at 0.975 (`circles`/`moons` mcs5 leaf). This subtask establishes the structure and provisional bounds; 54.9 confirms/tightens them against the real float32 outputs.
<!-- SECTION:NOTES:END -->
