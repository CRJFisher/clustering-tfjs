---
id: TASK-54.2
title: Re-base HDBSCAN parity tolerances for float32 (strict labels, relaxed probabilities)
status: To Do
assignee: []
created_date: '2026-06-24'
updated_date: '2026-06-24'
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

- [ ] #1 Tie-free fixtures keep exact `labels_equivalent_with_noise` assertions unchanged
- [ ] #2 The tie-free per-point probability assertion is loosened from `toBeCloseTo(..., 6)` to a documented float32 bound (set from the measured front-half drift, ~1e-4, plus slack — e.g. ~1e-3)
- [ ] #3 Tie-bound MAE and `alignment_agreement` bounds are re-measured under the float32 front-half and set from observed values plus slack (the precomputed-cosine fixture at 0.150 is the watch item)
- [ ] #4 The test-header comment is rewritten to explain the float32 parity stance: why labels are exact but probabilities are bounded, replacing the float64 "exceeding the bound means re-measure not loosen" framing
- [ ] #5 The suite passes against the float64 baseline with the new bounds (loosening a bound must never make a currently-passing assertion fail)

<!-- AC:END -->

## Implementation Notes

Measured drift to anchor the bound: tie-free probability overshoot was ≈9.9e-5 (`nested_mcs8_ms2_*_eps0.8`) and ≈3.2e-5 (`blobs_overlap_mcs10_ms2_*`). Set the tie-free probability tolerance from the maximum observed once the front-half lands; this subtask establishes the _structure_ and a provisional bound, and 54.9 confirms/tightens it against the real float32 outputs. Keep the documentation-style rule: describe the current contract, not the history.
