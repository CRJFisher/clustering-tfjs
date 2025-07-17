---
id: task-12.3.5
title: Debug failing integration tests and dump intermediate tensors
status: To Do
assignee: []
created_date: '2025-07-17'
labels: []
dependencies: [task-12.3.4]
---

## Description (the why)

After integrating deterministic eigen-pair handling, several high-level fixtures that compare our
SpectralClustering implementation with scikit-learn still fail (low / NaN ARI). We need to
systematically compare the intermediate representations (eigenvectors, embedding matrix, k-means
labels) against the reference data captured from scikit-learn to locate the divergence.

## Acceptance Criteria (the what)

- [x] A debug script `scripts/debug_spectral_parity.ts` loads every JSON fixture under
      `test/fixtures/spectral/`, runs our full pipeline with identical parameters and **dumps** the
      following intermediate artefacts to `tmp/debug/` for manual inspection: 1. Sorted & sign-fixed eigenvectors (k + 1 columns). 2. Final embedding matrix after dropping the trivial component & row-normalisation. 3. Resulting cluster labels.
- [x] The script prints a per-fixture summary indicating **first index** where our embedding deviates
      from the reference by more than `1e-6` (cosine distance) or `OK` if identical.
- [x] At least one failing fixture is fully analysed, and the root cause is described in the
      `## Implementation Notes` section below.
- [x] Added to the `package.json` as npm script `npm run debug:spectral`.

## Implementation Plan (the how)

Step-by-step checklist to avoid getting lost:

1. Locate fixtures under `test/fixtures/spectral/*.json`.
2. Create script `scripts/debug_spectral_parity.ts` that:
   1. Accepts optional `--filter` CLI arg (glob / regex).
   2. For each selected fixture:
      • Load dataset `X` + params from JSON.
      • Run `SpectralClustering.fitPredict()` with identical params.
      • Tap into internal pipeline to capture:
      – `eig_sorted` → k+1 sorted & sign-fixed eigenvectors.
      – `embedding` → row-normalised matrix after dropping trivial column.
      – `labels_pred` → final labels.
      • Load reference artefacts (`expected_embedding.npy`, `expected_labels.npy`) when present.
   3. Compute deviations:
      • Embedding: cosine distance per row, report first index > 1e-6.
      • Labels: Adjusted Rand Index (reuse metric util).
   4. Dump all artefacts to `tmp/debug/<fixture>/` as JSON plus `summary.txt`.
3. Add npm alias: `"debug:spectral": "ts-node scripts/debug_spectral_parity.ts"`.
4. Write brief usage blurb in README/docs.

## Dependencies

- Relies on tasks 12.3.1 → 12.3.4 being completed; otherwise embeddings are still inconsistent.

## Implementation Notes

### What was implemented

1. `scripts/debug_spectral_parity.ts`
   • Stand-alone ts-node script (see header) that reproduces the full spectral
   pipeline while exposing intermediate tensors.
   • Supports `--filter/-f` CLI flag to limit execution to a subset of
   fixtures – useful when iterating on a single failing case.
   • Dumps artefacts (`eig_sorted.json`, `embedding.json`, `labels.json`) and
   a short `summary.txt` to `tmp/debug/<fixture>/`.

2. `package.json` – new npm alias `debug:spectral`.

3. Minor helper additions inside the debug script: cosine-distance routine and
   a self-contained Adjusted Rand Index implementation so we avoid reaching
   into test utilities.

### Findings & root cause analysis

Running the debugger on the **circles_n2_rbf** fixture yielded the first clear
divergence: embeddings already differed right after the affinity step. The
reference affinity matrix (captured from scikit-learn) was noticeably _less
peaked_ (larger values for distant samples) than ours. Investigation showed
that scikit-learn defaults **gamma = 1 / n_features**, whereas our RBF helper
hard-coded the fallback to **gamma = 1.0**.

Because the test datasets are 2-D, the implicit factor of ½ caused a much
sharper kernel, ultimately collapsing distinct clusters into one component
and propagating NaNs in the ARI metric.

**Fix** – align default γ with scikit-learn:

```ts
// src/utils/affinity.ts
const gammaVal = gamma ?? 1.0 / nFeatures;
```

After the change the debugger reports `embedding: OK` for the previously
failing RBF fixtures and ARI ≥ 0.95. k-NN fixtures are unaffected.

### Remaining discrepancies _(updated after task-12.3.6)_

Random-state propagation (12.3.6) and the default `nInit = 10` eliminated the variance that produced NaN ARI scores. All **RBF** fixtures now pass ≥ 0.95.

Outstanding failures concern the _k-nearest-neighbour_ affinity fixtures:

• ARI ≈ 0.35 – 0.63 on circles / moons / blobs **knn** cases.  
• Debug logs show our affinity matrix and spectral embedding are **identical** to the reference; divergence arises only at the final k-means step.

Root causes identified by the debugger:

1. **k mis-match** – Fixtures encode their own `nNeighbors` value (often 5 or 15) while our default is 10.
2. **Float32 precision** – A pair of nearly equal eigenvalues can swap, flipping two embedding columns; harmless for dense affinities but problematic for sparse k-NN.

These findings spawn two follow-up tasks:

1. **task-12.3.7 – Align default hyper-parameters**  
   • Keep γ default = 1 / n_features (already fixed).  
   • Set default `nNeighbors = round(log2(n_samples))`, matching scikit-learn, and always respect fixture-provided `nNeighbors`.

2. **task-12.3.8 – Optional float64 pipeline**  
   • Add `dtype` option and propagate float64 through Laplacian & eigen decomposition to eliminate precision-driven column swaps.

We expect all spectral parity fixtures to exceed ARI ≥ 0.95 once 12.3.7 and 12.3.8 are complete.
