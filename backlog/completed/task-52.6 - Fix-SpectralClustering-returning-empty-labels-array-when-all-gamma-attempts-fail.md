---
id: TASK-52.6
title: >-
  Fix SpectralClustering returning empty labels array when all gamma attempts
  fail
status: Done
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

## Implementation Notes

## High-level summary

`intensive_parameter_sweep` is the optimization loop inside `SpectralClustering.fit` that tries multiple RBF gamma values and selects the best clustering by validation score. It initialised `best_result` with `{labels: []}` as a sentinel, and because NaN validation scores never satisfy `NaN > -Infinity`, any gamma producing a degenerate embedding silently left the sentinel in place. The function returned the empty-label result, `fit()` assigned it to `labels_`, and callers received silent garbage with no error.

The fix converts the sentinel pattern into an explicit contract: after sweeping all gammas, if `best_result.labels` is still empty the function throws a descriptive error. The per-gamma try block is also extended to cover the affinity-matrix and embedding computation steps, so a throwing embedding (e.g. an eigensolver failure) is treated as a skipped gamma rather than aborting the whole sweep with a raw eigendecomposition error. `SpectralClustering.fit` adds a try/catch around the `intensive_parameter_sweep` call to dispose the `U` and `x_tensor` tensors before rethrowing — without this, a failed sweep left live tensors with no way to reclaim them.

The per-gamma structure now reads: build affinity → compute embedding (with affinity disposal guaranteed in an inner finally) → score and compare → Phase B validation; the outer catch skips the gamma on any failure; the outer finally disposes the embedding. After the loop, the empty-labels check gates the return.

Tests cover three new cases: a unit test that drives `intensive_parameter_sweep` directly with a degenerate (all-zero) embedding and asserts the descriptive error message; a length-contract test that runs `fit_predict` through the `intensive_parameter_sweep` path with `gamma_range: [1.0]` on well-separated data; and the existing generic length-contract test on the standard path.

The `@throws` JSDoc on `intensive_parameter_sweep` documents the new failure contract for callers reading the signature.
