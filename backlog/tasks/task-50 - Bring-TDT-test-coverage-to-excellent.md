---
id: TASK-50
title: Bring TDT test coverage to excellent
status: To Do
assignee: []
created_date: '2026-06-07 08:35'
labels:
  - tdt
  - testing
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->

The Topic Detection and Tracking upgrade (task-49) is tested against scikit-learn reference outputs, but a coverage audit found branch-coverage gaps and a few behaviours not pinned to sklearn output. Raise the new TDT modules to excellent, sklearn-grounded coverage so correctness is locked in against regressions.

### Measured baseline (audit at task-49 completion)

| Module                | Stmts | Branch |
| --------------------- | ----- | ------ |
| kdistance             | 100%  | 100%   |
| minimum_spanning_tree | 98%   | 90%    |
| mutual_reachability   | 95%   | 80%    |
| cluster_tracking      | 97%   | 86%    |
| condensation_tree     | 95%   | 75%    |
| medoid_selection      | 93%   | 78%    |
| hdbscan               | 88%   | 74%    |
| pca                   | 94%   | 52%    |

Known gaps the audit surfaced:

- **Branch coverage** is low in `pca.ts` (52%), `hdbscan.ts` (74%), `condensation_tree.ts` (75%), `medoid_selection.ts` (78%) — uncovered: PCA error paths and the zero-norm fallback; HDBSCAN single-sample (`n===1`), precomputed non-square rejection, and the manhattan native path; the epsilon `traverse_upwards` branches; the medoid manhattan branch and empty-input path.
- **HDBSCAN estimator parity is tolerance-based** (cluster count exact, ≥95% label agreement, probability MAE ≤ 0.2) rather than exact, because mutual-reachability weight ties are ordered differently across implementations. The condensed-tree core is already bit-exact against sklearn's single-linkage hierarchy.
- **Not validated against actual sklearn output**: KMeans `metric:'cosine'` (only a synthetic sanity test), and degenerate HDBSCAN cases (all-noise input, single dense cluster).
- **Thin parameter sweeps**: `min_samples` (one case) and `cluster_selection_epsilon` (one value).
- No coverage threshold is enforced in CI for these modules, so regressions can erode coverage silently.

`exemplar_indices_` (HDBSCAN) and `medoid_indices_` have no scikit-learn equivalent to assert against; the goal for those is exhaustive behavioural coverage and documentation that they are library-defined, not sklearn-derived.

<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->

- [ ] Branch coverage for every new TDT module (`condensation_tree`, `minimum_spanning_tree`, `mutual_reachability`, `kdistance`, `hdbscan`, `cluster_tracking`, `medoid_selection`, `pca`, `representations`) is ≥ 90%, and statement/line coverage is ≥ 95%
- [ ] The previously-uncovered branches are exercised: PCA error paths + zero-norm fallback; HDBSCAN `n===1` single-sample, precomputed non-square rejection, and manhattan native path; the epsilon `traverse_upwards` branches; medoid manhattan metric branch and empty-input path
- [ ] HDBSCAN has scikit-learn reference fixtures for the degenerate cases it must handle: an all-noise dataset (every label `-1`) and a single dense cluster, with labels and `probabilities_` asserted against sklearn
- [ ] HDBSCAN `min_samples` and `cluster_selection_epsilon` are each swept over at least three values against sklearn fixtures, in both `eom` and `leaf` modes
- [ ] The HDBSCAN probability assertion is tightened: the per-point probability tolerance is documented and as tight as the tie-ambiguity allows (e.g. exact for tie-free fixtures, a stated bound otherwise) rather than a blanket MAE ≤ 0.2
- [ ] KMeans `metric:'cosine'` (spherical k-means) is validated against a scikit-learn reference fixture (`normalize(X)` + `KMeans`), asserting centroids and labels up to permutation
- [ ] `exemplar_indices_` and `medoid_indices_` are documented in code as library-defined (no scikit-learn equivalent) and covered by behavioural tests (correct count, valid in-range indices, deterministic tie-breaking)
- [ ] CI enforces a coverage threshold for the TDT modules so future regressions fail the build
- [ ] All fixtures are regenerated reproducibly by the `tools/sklearn_fixtures/*.py` generators; no committed fixture is hand-edited
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->

1. Run `jest --coverage` scoped to the TDT modules to get the current per-file branch report and the exact uncovered line numbers.
2. Add targeted edge-case tests to hit uncovered branches (PCA errors/zero-norm; HDBSCAN n=1 / precomputed-non-square / manhattan; medoid manhattan/empty; epsilon traverse_upwards).
3. Extend `generate_hdbscan.py` with all-noise and single-cluster datasets and wider `min_samples`/`cluster_selection_epsilon` sweeps; regenerate fixtures.
4. Add a `generate_kmeans.py` cosine case (`normalize(X)` + euclidean `KMeans`) and a colocated spherical-k-means parity test.
5. Tighten the HDBSCAN probability assertion: assert exact for tie-free fixtures; keep a documented bound only where ties genuinely diverge.
6. Add a coverage threshold (jest `coverageThreshold`) for the TDT paths and wire it into CI.
7. Document `exemplar_indices_` / `medoid_indices_` as library-defined representatives.
<!-- SECTION:PLAN:END -->
