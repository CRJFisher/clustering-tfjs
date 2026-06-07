---
id: TASK-49.10
title: Add JSON serialization to KMeans with predict-after-restore
status: Done
assignee: []
created_date: '2026-06-05'
updated_date: '2026-06-06 12:22'
labels:
  - tdt
  - deferred
dependencies:
  - task-49.6
parent_task_id: TASK-49
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->

A fitted KMeans model can be persisted as JSON and rehydrated so callers store a trained model and later assign new samples to its clusters without re-fitting. The fitted state that fully determines cluster assignment is the centroid matrix; together with the constructor params and the inertia criterion it forms a complete, self-describing snapshot. Restoring that snapshot reproduces cluster assignment exactly, which requires a predict path over the restored centroids.

Serialization exists only where it is meaningful. SpectralClustering and AgglomerativeClustering are transductive: their labels come from a graph embedding or a linkage hierarchy built over the specific input set, with no centroid or parametric model that can assign an unseen point. They expose no predict and no JSON round-trip, and the public surface documents this so the asymmetry reads as an intentional, explained property rather than a gap.

The snapshot stores the live fitted state directly: the centroid matrix, the constructor params, and the inertia value, with the same field names used in memory.

<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->

- [x] #1 A fitted KMeans exposes `to_json()` returning a plain JSON-serializable object containing the `centroids_` matrix, the constructor params, and `inertia_`
- [x] #2 `KMeans.from_json()` reconstructs a KMeans whose `centroids_`, params, and `inertia_` equal those of the original fitted instance
- [x] #3 KMeans exposes `predict(X: DataMatrix): Promise<number[]>` assigning each sample to its nearest restored centroid
- [x] #4 `predict()` on a model restored via `from_json()` reproduces the original fitted model's cluster assignment exactly for the same `X`, verified by a colocated test in `src/clustering/kmeans.test.ts`
- [x] #5 SpectralClustering and AgglomerativeClustering provide neither `predict()` nor `to_json()`/`from_json()`, asserted by tests in `src/clustering/spectral.test.ts` and `src/clustering/agglomerative.test.ts`
- [x] #6 `docs/API.md` states that KMeans supports predict and JSON serialization while SpectralClustering and AgglomerativeClustering do not, and explains that these estimators are transductive with no model that can assign unseen samples
- [x] #7 `to_json()`, `from_json()`, and `predict()` use no `as any`, `as unknown`, or `as never` assertions
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->

KMeans to*json/from_json (centroids*, params, inertia\_) with predict-after-restore exact. Spectral/Agglomerative assert no predict/serialization (transductive). docs/API.md documents the asymmetry.

<!-- SECTION:NOTES:END -->
