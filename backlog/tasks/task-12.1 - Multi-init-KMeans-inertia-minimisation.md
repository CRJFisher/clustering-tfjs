# task-12.1 - Multi-init K-Means inertia minimisation

## Description (the why)

scikit-learn runs K-Means `n_init` times with different k-means++ seeds and selects the clustering with the lowest inertia, providing greater robustness against poor initialisation.  Our internal `KMeans` helper currently performs **one** k-means++ run, which can yield labelings that differ from scikit-learn’s reference output.  Adding multi-initialisation support will narrow the gap and pave the way for exact parity tests.

## Acceptance Criteria (the what)

- [ ] `KMeansParams` gains optional `nInit` (int ≥ 1, default 10).
- [ ] `KMeans` runs k-means++ initialisation `nInit` times, keeps the solution with the smallest inertia.
- [ ] Behaviour is deterministic when `randomState` is supplied (each init must derive its own PRNG stream from the base seed).
- [ ] Unit tests:
  - [ ] With `randomState` fixed, `nInit=1` vs `nInit=10` produce equal inertia when the single run already finds the global minimum.
  - [ ] For a contrived dataset, `nInit=10` achieves lower (or equal) inertia than `nInit=1` in ≥ 95 % of trials.
- [ ] Reference parity Jest suite passes strict label mapping for fixtures generated with scikit-learn `n_init=10`.

## Implementation Plan (the how)

1. Extend `KMeansParams` type; update validation logic.
2. Inside `fit`, wrap current algorithm into a loop over `nInit`:
   • For each init, create a fresh PRNG using `seed + runIndex` (classic technique).
   • Run k-means++, Lloyd iterations → inertia.
   • Track best centroids / labels / inertia.
3. After loop, set public properties to best solution.
4. Expose `inertia_` publicly as before.
5. Update docs & JSDoc.
6. Add/update tests as per AC.

## Dependencies

Relates to: task-12 (SpectralClustering core parity).

## Implementation Notes (to fill after completion)

*TBD*

