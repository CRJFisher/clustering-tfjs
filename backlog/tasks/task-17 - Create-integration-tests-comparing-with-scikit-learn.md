---
id: task-17
title: Create integration tests comparing with scikit-learn
status: To Do
assignee: []
created_date: '2025-07-15'
labels: []
dependencies:
  - task-16
---

## Description

Develop comprehensive integration tests that validate the TypeScript implementations against scikit-learn outputs using various synthetic and real-world datasets

## Implementation Plan

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

## Acceptance Criteria

- [ ] Test datasets prepared (blobs, moons, circles)
- [ ] Python scripts generate expected outputs for all algorithms
- [ ] Integration tests for AgglomerativeClustering with all linkage types
- [ ] Integration tests for SpectralClustering with both affinity types
- [ ] Integration tests for all validation metrics
- [ ] Tolerance thresholds defined for numerical differences
- [ ] Edge case tests (empty data, single cluster, outliers)
- [ ] Performance comparison reports generated
