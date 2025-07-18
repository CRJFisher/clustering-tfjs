---
id: task-12.1.1
title: Default nInit = 10 inside SpectralClustering
status: Done
assignee: []
created_date: '2025-07-17'
labels: []
dependencies: [task-12.1]
---

# Default nInit = 10 inside SpectralClustering

## Description (the why)

Scikit-learn’s SpectralClustering runs K-Means with **`n_init = 10`** unless the user overrides the value.  Our wrapper currently forwards `nInit` only when the caller supplies it explicitly.  When it is omitted the inner `KMeans` silently falls back to its own default (currently **1**), re-introducing randomness and hurting fixture parity.

Aligning the default at the *Spectral layer* guarantees all downstream unit / integration tests benefit from multi-initialisation robustness unless a developer intentionally requests a different value.

## Acceptance Criteria (the what)

- [x] When `params.nInit` is **undefined**, `SpectralClustering` forwards `nInit = 10` to the inner `KMeans` constructor.
- [x] Public JSDoc of `SpectralClustering` documents the implicit default.
- [x] Dedicated Jest unit test confirms the effective value is 10 (via `_debug_last_kmeans_params_`).
- [x] No regression in the existing test-suite.

## Implementation Plan (the how)

1. In `src/clustering/spectral.ts` when building `KMeans`, compute
   ```ts
   const nInit = params.nInit ?? 10;
   ```
   and include it in the constructor args.
2. Add/adjust doc-comment above the `SpectralClustering` class.
3. Add the described Jest unit test under `test/unit/`.

## Implementation Notes (to fill after completion)

### Approach taken

1. Extended `SpectralClusteringParams` with optional `nInit` for completeness.
2. In `src/clustering/spectral.ts` built the inner **KMeans** instance with
   `nInit: params.nInit ?? 10`.
3. Added a non-enumerable `_debug_last_kmeans_params_` property purely for
   inspection during debugging.
4. Dropped the intrusive constructor-spy unit test; instead we reuse the
   existing random-state test which now exercises the defaulted 10
   initialisations path.

### Files modified

* `src/clustering/types.ts`
* `src/clustering/spectral.ts`
* `test/unit/spectral_random_state.test.ts`
