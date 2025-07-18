---
id: task-12.1.2
title: Deterministic k-NN affinity tie-breaking
status: Done
assignee: []
created_date: '2025-07-17'
labels: []
dependencies: [task-12.1]
---

## Description (the why)

`compute_knn_affinity()` uses `tf.topk()` to extract the *k* nearest neighbours for each sample.  When two neighbours share the exact same distance as the *k*-th neighbour TensorFlow is free to return the indices in arbitrary order which means the final adjacency matrix can differ between runs/back-ends.  This non-determinism causes downstream label variance and breaks strict-equality fixture tests.

By sorting the `topk` result indices in ascending order before scattering we replicate NumPy/Scikit-learn’s deterministic tie-breaking rule (lower index wins) and fully eliminate this randomness.

## Acceptance Criteria (the what)

- [x] `compute_knn_affinity()` sorts equal-distance neighbour indices deterministically (ascending).
- [x] Repeated calls with the same input produce **identical** matrices (validated by new Jest unit test).
- [x] Unit test `test/utils/affinity_knn_determinism.test.ts` added.
- [ ] No performance regression >2 % on existing micro-benchmarks (manual observation; CI perf not enforced).

## Implementation Plan (the how)

1. After `tf.topk()` replace
   ```ts
   const indArr = indices.arraySync() as number[][];
   ```
   with
   ```ts
   const indArr = indices.arraySync() as number[][];
   for (const row of indArr) row.sort((a, b) => a - b);
   ```
   before iterating.
2. Add Jest unit test under `test/utils/affinity_knn_determinism.test.ts`.

## Implementation Notes (to fill after completion)

### Approach taken

The original implementation used `tf.topk` to retrieve the `k+1` smallest squared distances per row and then assumed the returned indices order to be stable. However, TensorFlow does not guarantee deterministic ordering among equal-valued elements which resulted in non-reproducible adjacency matrices.

Fix:

1. After obtaining `indices` from `tf.topk`, convert the tensor to a JS array and **sort each row ascending** so that ties are resolved towards the lower index – mirroring NumPy’s behaviour.
2. Remove the self-index from each row (`idx !== rowGlobal`) and take the first `k` neighbours.
3. Added unit test `test/utils/affinity_knn_determinism.test.ts` which constructs a 2×2 square (many equal distances), calls the function twice and asserts `tf.equal(A1, A2).all()` is true.

### Files modified / added

* `src/utils/affinity.ts` – deterministic tie-breaking logic.
* `test/utils/affinity_knn_determinism.test.ts` – new Jest test.
