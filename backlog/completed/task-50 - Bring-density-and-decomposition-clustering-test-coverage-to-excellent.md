---
id: TASK-50
title: Bring density and decomposition clustering test coverage to excellent
status: Done
assignee:
  - '@claude'
created_date: '2026-06-07 08:35'
updated_date: '2026-06-09 20:56'
labels:
  - testing
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The density, cosine, and representation clustering upgrade (task-49) is tested against scikit-learn reference outputs, but a coverage audit found branch-coverage gaps and a few behaviours not pinned to sklearn output. Raise these new clustering modules to excellent, sklearn-grounded coverage so correctness is locked in against regressions.

### Measured baseline (audit at task-49 completion)

| Module                | Stmts | Branch |
| --------------------- | ----- | ------ |
| kdistance             | 100%  | 100%   |
| minimum_spanning_tree | 98%   | 90%    |
| mutual_reachability   | 95%   | 80%    |
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
- [x] #1 Branch coverage for every new clustering module (`condensation_tree`, `minimum_spanning_tree`, `mutual_reachability`, `kdistance`, `hdbscan`, `medoid_selection`, `pca`) is ≥ 90%, and statement/line coverage is ≥ 95%. (`representations` is type-only — a single interface with no runtime code — so it cannot carry an instrumented threshold; its contract is covered behaviourally by the estimator tests.)
- [x] #2 The previously-uncovered branches are exercised: PCA error paths + zero-norm fallback (extracted into the directly-testable `unit_init_vector` helper); HDBSCAN `n===1` single-sample, precomputed non-square rejection, and manhattan native path; the epsilon `traverse_upwards` branches; medoid manhattan metric branch and empty-input path
- [x] #3 HDBSCAN has scikit-learn reference fixtures for the degenerate cases it must handle: an all-noise dataset (every label `-1`) and a single dense cluster, with labels and `probabilities_` asserted against sklearn. (With `allow_single_cluster=False` — the only mode the estimator supports — HDBSCAN structurally cannot emit exactly one cluster, since candidate clusters are born in sibling pairs; sklearn's reference output for the single-dense-blob input is therefore itself all-noise, and that is what the fixture pins.)
- [x] #4 HDBSCAN `min_samples` and `cluster_selection_epsilon` are each swept over at least three values against sklearn fixtures, in both `eom` and `leaf` modes
- [x] #5 The HDBSCAN probability assertion is tightened: the per-point probability tolerance is documented and as tight as the tie-ambiguity allows (exact to 1e-6 for tie-free fixtures, a stated MAE ≤ 0.16 bound otherwise) rather than a blanket MAE ≤ 0.2. (Tie-free exactness required moving HDBSCAN's native-metric distances from the float32 TensorFlow.js path to plain-JS float64.)
- [x] #6 KMeans `metric:'cosine'` (spherical k-means) is validated against a scikit-learn reference fixture (`normalize(X)` + `KMeans`), asserting centroids and labels up to permutation
- [x] #7 `exemplar_indices_` and `medoid_indices_` are documented in code as library-defined (no scikit-learn equivalent) and covered by behavioural tests (correct count, valid in-range indices, deterministic tie-breaking)
- [x] #8 CI enforces a coverage threshold for these clustering modules so future regressions fail the build
- [x] #9 All fixtures are regenerated reproducibly by the `tools/sklearn_fixtures/*.py` generators; no committed fixture is hand-edited
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Run `jest --coverage` scoped to these clustering modules to get the current per-file branch report and the exact uncovered line numbers.
2. Add targeted edge-case tests to hit uncovered branches (PCA errors/zero-norm; HDBSCAN n=1 / precomputed-non-square / manhattan; medoid manhattan/empty; epsilon traverse_upwards).
3. Extend `generate_hdbscan.py` with all-noise and single-cluster datasets and wider `min_samples`/`cluster_selection_epsilon` sweeps; regenerate fixtures.
4. Add a `generate_kmeans.py` cosine case (`normalize(X)` + euclidean `KMeans`) and a colocated spherical-k-means parity test.
5. Tighten the HDBSCAN probability assertion: assert exact for tie-free fixtures; keep a documented bound only where ties genuinely diverge.
6. Add a coverage threshold (jest `coverageThreshold`) for these module paths and wire it into CI.
7. Document `exemplar_indices_` / `medoid_indices_` as library-defined representatives.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
### High-level summary

The eight density/decomposition modules from task-49 now sit at 100% statement/line/function coverage with branch coverage between 93% and 100% (baseline worst case: `pca.ts` at 52% branch), enforced in CI by a per-file jest gate. The headline structural change is in how HDBSCAN parity is asserted: every fixture records a `tie_free` flag, computed in the generator from the gaps between sorted mutual-reachability MST edge weights. Distinct weights make the MST — and hence the whole flat clustering — mathematically unique, so tie-free fixtures are asserted exactly (labels up to permutation, probabilities to 1e-6), while tie-bound fixtures keep a documented MAE ≤ 0.16 bound (observed worst: 0.150 on the heavily-tied precomputed-cosine fixture; every other fixture ≤ 0.077). Making the exact tier honest required moving HDBSCAN's native-metric distance computation from the float32 TensorFlow.js path to plain-JS float64 — the rest of the pipeline (MST, condensed tree) is already float64, and the float32 round-trip was the only thing standing between the implementation and bit-level agreement with scikit-learn. After that change most previously "tolerance-only" fixtures match sklearn exactly.

### Key decisions

- **Single-dense-cluster semantics.** With `allow_single_cluster=False` (the only mode the estimator supports), HDBSCAN structurally cannot emit exactly one cluster: candidate clusters are created in sibling pairs during condensation, so a lone dense blob is root-only and comes back all-noise. The `single_blob` fixture pins sklearn's actual output (all `-1`) rather than a wished-for single label; AC #3 carries the reconciliation.
- **`representations.ts` carve-out.** The module is a single interface — type-only, no runtime code — so jest cannot instrument it and a per-file threshold would hard-fail with "coverage data not found". It is excluded from the gate and covered behaviourally through the estimator tests; documented in `jest.coverage.config.js`, CONTRIBUTING.md, and AC #1.
- **Deliberate sklearn deviations pinned, not changed.** HDBSCAN returns all-noise for a single sample and clamps `min_samples` to `n` where sklearn raises in both cases. Both behaviours predate this task; they are now documented as intentional in code and `docs/API.md` and pinned by tests.
- **PCA zero-norm fallback.** The branch is unreachable through the real RNG (it requires a randomly-drawn all-zero vector), so the fallback was extracted into the exported `unit_init_vector` helper and tested directly — a behaviour-preserving testability refactor, not an RNG mock.
- **Epsilon sweep dataset.** `cluster_selection_epsilon` only changes the selection when the condensed tree is nested, so the sweep runs on a purpose-built two-level blob hierarchy (two pairs of close blobs) where each swept value visibly coarsens the result (5 → 3 → 2 clusters under eom); the generator asserts non-vacuity so drift fails loudly.
- **Coverage gate design.** `GATED_MODULES` in `jest.coverage.config.js` is the single source of truth: it drives instrumentation, per-file thresholds (branch ≥ 90, stmts/lines/funcs ≥ 95 — the AC floor, with measured values comfortably above), and test selection (colocated suites derived by convention). The gate runs on one CI cell (~16 s). The vestigial codecov upload step — which had referenced a never-generated lcov file since its introduction — was removed.

### Modified or added files

- `src/clustering/hdbscan.ts` — float64 native-metric distances, rectangular-input validation, documented deviations, `exemplar_indices_` TSDoc.
- `src/decomposition/pca.ts` — `unit_init_vector` extraction.
- `src/clustering/{hdbscan,medoid_selection,kmeans}.test.ts`, `src/decomposition/pca.test.ts`, `src/graph/{condensation_tree,minimum_spanning_tree,mutual_reachability}.test.ts` — tiered parity assertions, degenerate-case tests, hand-built condensed-tree epsilon tests, metric-behaviour and tie-determinism tests, cosine k-means parity.
- `test_support/label_agreement.ts` — shared `labels_equivalent_with_noise` / `alignment_agreement` helpers (deduplicated from two test files).
- `tools/sklearn_fixtures/generate_hdbscan.py` — `tie_free`/`min_mst_gap` recording, degenerate datasets, min_samples and epsilon sweeps, loud generation asserts; `generate_kmeans.py` — cosine (`normalize(X)` + KMeans) case.
- `__fixtures__/hdbscan/` (33 fixtures, 16 new), `__fixtures__/kmeans/cosine_directions_n3.json`.
- `jest.coverage.config.js` (new), `package.json` (`test:coverage:gate`), `.github/workflows/ci.yml`, `CONTRIBUTING.md`, `docs/API.md`, `src/clustering/representations.ts`, `src/clustering/medoid_selection.ts` (docs).

Verification: full suite 942/942 green; `npm run test:coverage:gate` passes with every gated module ≥ 93% branch and ≥ 97% statements; lint and type-check clean. Reviewed by a 10-lens agent panel; fixes applied for the ragged-input regression in the rewritten distance path, helper duplication, gate config duplication, and the codecov scope concern.
<!-- SECTION:NOTES:END -->
