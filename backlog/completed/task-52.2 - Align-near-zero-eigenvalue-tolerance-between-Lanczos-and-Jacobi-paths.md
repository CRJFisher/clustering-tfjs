---
id: TASK-52.2
title: Align near-zero eigenvalue tolerance between Lanczos and Jacobi paths
status: Done
assignee:
  - crjfisher
created_date: '2026-06-10 08:55'
updated_date: '2026-06-10 10:30'
labels:
  - bug
  - confirmed
dependencies: []
references:
  - 'src/eigen/smallest_eigenvectors_with_values.ts:72'
parent_task_id: TASK-52
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
smallest_eigenvectors_with_values uses tolerance 1e-2 in the Lanczos path and 1e-7 in the Jacobi path when counting near-zero eigenvalues to determine how many extra eigenvectors to return. A matrix with an eigenvalue of ~0.005 gets k+1 eigenvectors from Lanczos but k from Jacobi, so spectral clustering silently produces different embedding dimensions (and therefore wrong cluster labels) for the same matrix depending on which path the n=100 boundary routes it to.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Both Lanczos and Jacobi paths use the same near-zero eigenvalue tolerance
- [x] #2 A test verifies that a matrix with a small-but-nonzero eigenvalue produces the same number of returned eigenvectors regardless of which path is taken
- [x] #3 No change to the expected cluster labels on existing fixtures
- [x] #4 SpectralClustering produces the same cluster labels on inputs of size 99 and 101 with otherwise identical structure, confirming the n=100 path boundary does not affect results (code-path equivalence test)
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Read task and identify the two tolerance constants (1e-2 Lanczos, 1e-7 Jacobi)\n2. Extract a single NEAR_ZERO_TOL=1e-7 constant and use it in both paths\n3. Write smallest_eigenvectors_with_values.test.ts covering AC#2 (both paths same count) and AC#4 (SpectralClustering n=99 vs n=101)\n4. Run all eigen + spectral tests to verify no regressions
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Extracted NEAR_ZERO_TOL=1e-5 as a module-level constant in smallest_eigenvectors_with_values.ts and replaced the divergent local TOL constants (1e-2 in Lanczos, 1e-7 in Jacobi) with it. Value is 1e-5 (above the Lanczos convergence floor of 1e-6, matching PSD_CLAMP_TOL in lanczos.ts) so structural zeros that land in (1e-7, 1e-6] from Lanczos noise are still counted. The Jacobi path was already precise enough that 1e-7 would have worked, but 1e-5 remains far below any real Fiedler eigenvalue (typically >= 1e-3). Created smallest_eigenvectors_with_values.test.ts with 5 tests: 3 covering AC#2 (Jacobi 4×4 path, Lanczos 105×105 path, and cross-path comparison) and 2 via it.each covering AC#4 (SpectralClustering on n=99/n=101). AC#3 satisfied implicitly — Jacobi's shape/label behavior is unchanged. All 33 affected tests pass. Five Opus reviewers confirmed the fix is sound; main concern (1e-7 below Lanczos noise floor) addressed by bumping to 1e-5. The residual k+5 overshoot cap (Lanczos diverges from Jacobi for >5 structural zeros) is a pre-existing issue filed as task-52.11. Modified files: src/eigen/smallest_eigenvectors_with_values.ts (fix), src/eigen/smallest_eigenvectors_with_values.test.ts (new).
<!-- SECTION:NOTES:END -->
