---
id: TASK-17
title: Create integration tests comparing with scikit-learn
status: To Do
assignee: []
created_date: '2025-07-15'
updated_date: '2026-06-07 08:32'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Develop comprehensive integration tests that validate the TypeScript implementations against scikit-learn outputs using various synthetic and real-world datasets
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Test datasets prepared (blobs, moons, circles)
- [ ] #2 Python scripts generate expected outputs for all algorithms
- [ ] #3 Integration tests for AgglomerativeClustering with all linkage types
- [ ] #4 Integration tests for SpectralClustering with both affinity types
- [ ] #5 Integration tests for all validation metrics
- [ ] #6 Tolerance thresholds defined for numerical differences
- [ ] #7 Edge case tests (empty data, single cluster, outliers)
- [ ] #8 Performance comparison reports generated
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
The repository now contains a dedicated sub-project under
`tools/sklearn_fixtures` for producing reference outputs from scikit-learn.

Steps to extend / refresh fixtures:

1. `cd tools/sklearn_fixtures`
2. `python -m venv .venv && source .venv/bin/activate`
3. `pip install -r requirements.txt`
4. `python generate.py --out-dir ../../test/fixtures/agglomerative` (or another
   directory for metric fixtures)

New JSON files will automatically be picked up by the parity Jest tests (see
`test/clustering/*_reference.test.ts`).  When implementing additional
algorithms simply extend `generate.py` with extra parameter grids and output
directories.
<!-- SECTION:PLAN:END -->

## Implementation Notes

Archived as superseded. The sklearn-parity fixture pipeline this task describes
is fully built and operational, delivered incrementally by other tasks:

- AC #1, #3, #6 — Agglomerative parity: `src/clustering/agglomerative_reference.test.ts`
  runs exact (permutation-invariant) label matching over 13 fixtures in
  `__fixtures__/agglomerative/`.
- AC #4 — Spectral parity: `src/clustering/spectral_reference.test.ts` (ARI ≥ 0.95
  over 14 fixtures, both `rbf` and `knn` affinity) plus
  `spectral_embedding_numerical.test.ts` (intermediate embedding vs sklearn).
- AC #2 — Reference generators exist under `tools/sklearn_fixtures/`
  (`generate.py`, `generate_spectral.py`, `generate_spectral_embedding.py`,
  `generate_som.py`).
- AC #5 — Validation metrics (ARI, NMI, Calinski-Harabasz, Davies-Bouldin) carry
  inline sklearn reference values in their colocated `*.test.ts` files.
- AC #7 — Edge cases (single sample, n_clusters == n_samples, coincident points)
  covered in `agglomerative.test.ts` and `kmeans.test.ts`.

Remaining items are not worth keeping the task open:

- AC #8 (sklearn-vs-TS performance comparison reports) — YAGNI; never part of the
  codebase intent.
- Fixture-driven (vs inline) validation-metric tests and a KMeans parity fixture
  set are marginal gaps; if pursued, they belong in a focused, narrowly-scoped
  task rather than this broad umbrella.
