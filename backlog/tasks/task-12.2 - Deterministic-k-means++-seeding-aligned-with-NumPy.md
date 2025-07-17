# task-12.2 - Deterministic k-means++ seeding aligned with NumPy

## Description (the why)

K-means++ seeding relies on randomness when choosing the first centroid and when sampling the subsequent probability-weighted candidates.  scikit-learn implements this sampling using NumPy’s `RandomState` / `Generator` stream, which—given the **same integer seed**—always produces the same initialisation sequence.  

Our TypeScript implementation currently delegates all RNG needs to a small utility based on `Math.random()` or a simple LCG.  This means that, even if the user passes the same `randomState` value as in scikit-learn, the centroids chosen by k-means++ will differ, leading to divergent final cluster labels.  

Providing *bit-for-bit* deterministic seeding that mimics NumPy’s random stream will allow us to reproduce scikit-learn’s reference labelings exactly (in combination with tasks 12.1, 12.3 and 12.4).

## Acceptance Criteria (the what)

- [ ] Introduce a portable, seedable RNG that matches NumPy’s legacy `MT19937` algorithm (e.g. via a small, self-contained TypeScript port).
- [ ] `kMeansPlusPlusInit` uses this RNG whenever a user supplies `randomState` (falls back to current fast RNG when `randomState` is `undefined`).
- [ ] For a fixed dataset and `randomState = 42`, the initial centroids returned by our implementation are identical (within `1e-12` for float64) to those returned by `sklearn.cluster.k_means_._kmeans_plusplus` in Python 3.12 / scikit-learn 1.5.
- [ ] Unit test compares our centroid indices to a fixture generated offline with Python + NumPy (checked into `test/fixtures/`).
- [ ] Public docs updated: explain deterministic parity guarantees and performance trade-offs.

## Implementation Plan (the how)

1. Add minimal MT19937 implementation under `src/utils/rng/mt19937.ts` (existing OSS implementations can be referenced but re-implemented to avoid licence issues).
2. Create `RandomStream` interface with methods `rand()` and `randInt(max)` to abstract RNG choice.
3. Refactor current k-means++ helper to inject a `RandomStream`; adapt call sites in `KMeans` and `SpectralClustering`.
4. On the test side, generate fixtures using a small Python script (already committed) and assert equality.
5. Update `README` and relevant JSDoc.

## Dependencies

Relates to task-12 (SpectralClustering parity) and complements task-12.1 (multi-initialisation).

## Implementation Notes (to fill after completion)

*TBD*

