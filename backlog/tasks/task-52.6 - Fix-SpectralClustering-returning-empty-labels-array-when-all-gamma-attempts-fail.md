---
id: TASK-52.6
title: >-
  Fix SpectralClustering returning empty labels array when all gamma attempts
  fail
status: To Do
assignee: []
created_date: '2026-06-10 08:55'
labels:
  - bug
  - confirmed
dependencies: []
references:
  - 'src/clustering/spectral_optimization.ts:129'
parent_task_id: TASK-52
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->

intensive*parameter_sweep initialises best_result with an empty labels array. If all gamma attempts produce degenerate embeddings (NaN validation scores that never beat the -Infinity baseline), the inner catch blocks swallow the failures and the function returns {labels: []}. SpectralClustering.fit() assigns this directly to labels* without checking length, and fit_predict's null-check does not catch the empty-array case, so the caller receives silent garbage output.

<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->

- [ ] #1 When all gamma attempts fail to produce a valid score, intensive_parameter_sweep throws a descriptive error rather than returning an empty-label sentinel
- [ ] #2 SpectralClustering.fit() propagates that error to the caller
- [ ] #3 A test verifies that degenerate inputs (all-NaN scores) produce an error rather than empty labels
- [ ] #4 A test asserts that fit_predict always returns an array whose length equals n_samples (output contract test — catches any silent empty-result regression)
<!-- AC:END -->
