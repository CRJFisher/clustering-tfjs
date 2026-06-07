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
