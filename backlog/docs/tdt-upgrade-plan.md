# Topic Detection and Tracking (TDT) Upgrade Plan for clustering-tfjs

> **Path note (read first):** This plan was produced against an earlier `src/utils/` layout.
> The repo has since been refactored, so remap proposed file locations as follows:
>
> - `src/utils/pairwise_distance.ts` → `src/distance/pairwise_distance.ts`
> - cosine affinity / `compute_*_affinity` → `src/graph/affinity.ts` (extend) or `src/distance/`
> - MST / mutual-reachability / k-distance / condensed-tree → new files under `src/graph/`
>   (alongside `connected_components.ts`, `laplacian.ts`)
> - `findOptimalClusters` / `kneedle` → `src/model_selection/`
> - `tf-adapter` import → `src/backend/adapter`
> - new estimators (`hdbscan.ts`, etc.) → `src/clustering/`
>   The _substance_ (sequencing, fixture strategy, API design, hard-parts, backlog tasks) is unaffected.

**Audience:** the bergamot browsing-timeline consumer and the clustering-tfjs maintainers.
**Status:** decisive implementation plan, ready to decompose into Backlog.md tasks.

---

## 1. Executive Summary

Topic Detection and Tracking over a browsing timeline is, at its core, two distinct problems:

1. **Topic detection** — given a window of embedded browsing events, find coherent topics _without knowing how many there are_, and flag noise (incidental, one-off pages). This is a density-clustering problem on embedding vectors, where the natural metric is cosine.
2. **Topic tracking** — given clusters from consecutive windows, decide which topic at window _t_ _is_ which topic at window _t−1_, and emit `emerge / persist / merge / split / die` transitions with stable lifeline identities. This is the _only genuinely temporal_ feature in the entire backlog.

Everything else (PCA, MiniBatchKMeans, sample weights, predict/serialize, representation accessors, UMAP) is supporting infrastructure or convenience.

### The minimum viable bundle for TDT

| Capability                                                         | Why it is non-negotiable for TDT                                                                                                                                         | Item      |
| ------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | --------- |
| **HDBSCAN**                                                        | The field-default topic detector (BERTopic's default). Discovers variable-density topics, no `nClusters` required, emits noise for incidental pages.                     | Tier 1 #1 |
| **Cosine as a first-class metric**                                 | Browsing-event embeddings are direction-meaningful, magnitude-noisy. Cosine is the correct topic-similarity metric; Euclidean on raw embeddings clusters poorly.         | Tier 1 #2 |
| **Cross-window cluster tracking**                                  | The actual "Tracking" in Topic Detection and **Tracking**. Without it you have repeated snapshots, not a timeline.                                                       | Tier 3 #6 |
| **Cluster-representation accessors** (centroids/medoids/exemplars) | Tracking matches topics by their representative vectors. Tracking is dead-on-arrival without a uniform way to get a topic's representative point out of every estimator. | Tier 3 #8 |

That is the bundle. **DBSCAN, predict()+serialize, PCA, MiniBatchKMeans, and sample-weights are all valuable but each is a deferrable optimization or convenience, not a TDT enabler.** UMAP should be left external (see §6).

### The single genuinely-temporal feature

**`trackClusters(prev, curr, {threshold})`** (Tier 3 #6). It is the only item whose semantics are inherently about _time_. We elevate it from "Tier 3" to the second-highest priority — but it has a hard practical dependency on representation accessors (Tier 3 #8) and is most useful once HDBSCAN exists as the snapshot producer.

### Opinionated cuts

- **Defer DBSCAN/OPTICS.** Proposed as a "stepping stone toward HDBSCAN," but once we commit to HDBSCAN the stepping stone is wasted motion (YAGNI). HDBSCAN's k-distance / mutual-reachability machinery _subsumes_ DBSCAN. Add DBSCAN later only if a consumer explicitly needs fixed-`eps` semantics.
- **Leave UMAP external.** Non-deterministic, XL effort, not in the venv, fixtures can't be asserted by value.
- **Spectral/Agglomerative `predict()` is not provided.** There is no principled predict for these without full recompute. Don't fake it.

---

## 2. Sequenced Roadmap

Ordering is driven by three hard dependencies:

1. Tracking needs representation accessors (it matches by centroid/medoid/exemplar).
2. HDBSCAN's exemplar accessor is most naturally built alongside HDBSCAN, but the _generic_ centroid/medoid accessors are independent and unblock tracking for KMeans/Agglomerative/SOM immediately.
3. Cosine support is independent of all of the above and is a prerequisite for _good_ topic clustering and _good_ tracking.

### Phase 0 — Foundations & shared utilities (effort: M)

**Items:** the shared density/graph utilities that HDBSCAN needs, plus the `-1` noise-label design decision.

**Why first:** HDBSCAN is the largest item and depends on machinery the repo entirely lacks (MST, k-distance, mutual-reachability, condensed tree). Building and unit-testing these primitives in isolation de-risks the big estimator. The `-1` noise-label decision is a cross-cutting design choice that must be settled before any density estimator or validation metric is touched.

**Deliverables:**

- `kdistance.ts` — k-distance vector + core-distance extraction from a (block-wise) k-NN scan. Refactor the k-NN affinity builder to also expose `{neighbor_indices, neighbor_distances}` (root-cause fix: the function already computes these internally and throws them away).
- `mutual_reachability.ts` — `d_mreach(i,j) = max(core_k(i), core_k(j), d(i,j))`.
- `minimum_spanning_tree.ts` — Prim's algorithm over a dense distance/mutual-reachability matrix.
- A documented decision (`backlog/decisions/`) on noise labels: **density estimators emit `-1` for noise**; the repo invariant "no algorithm emits `-1`" is _amended_ to "non-density algorithms emit dense `0..k-1`; density algorithms emit `-1` for noise." Validation metrics must filter `-1` before computing pairwise distances.

**Test/fixture strategy:**

- Pure unit tests for MST (known small graphs, golden edge sets), k-distance, mutual-reachability — no sklearn needed; deterministic numeric primitives.
- Standalone cosine pairwise-distance fixture from `sklearn.metrics.pairwise_distances(metric='cosine')` to validate the cosine path independently.

**Effort:** M (~1–1.5 weeks).

---

### Phase 1 — Cosine as a first-class metric (Tier 1 #2) (effort: M)

**Items:** Cosine metric/affinity across Agglomerative (already validated, needs the affinity path wired), Spectral (`affinity='cosine'`), KMeans (`metric='cosine'` via L2-normalize = spherical k-means), and validation metrics gain an optional `metric` param.

**Why this order:** Independent of Phase 0; runs in parallel if staffed. Prerequisite for _quality_ in both HDBSCAN-on-embeddings and tracking. Lands before HDBSCAN so HDBSCAN's cosine path (via precomputed) reuses a tested cosine matrix builder.

**Deliverables:**

- `cosine_affinity.ts` — `compute_cosine_affinity(points)`.
- KMeans: `metric?: 'euclidean' | 'cosine'` + `normalize?: boolean`. Cosine = L2-normalize then Euclidean Lloyd (root-cause: refactor k-means++ seeding + Lloyd assignment to route through `pairwiseDistanceMatrix(X, metric)` rather than hardcoded squared-Euclidean).
- Spectral: `'cosine'` added to the `affinity` union, dispatching to `compute_cosine_affinity`.
- Validation metrics (`silhouette`, `calinski_harabasz`, `davies_bouldin`): optional `metric` parameter, default `'euclidean'`. **No compat shim** — update all internal callers to pass the metric explicitly.

**Test/fixture strategy (sklearn 1.3.2, already in venv):**

- Agglomerative `metric='cosine'`: direct sklearn fixtures, exact-up-to-permutation label parity.
- KMeans cosine: sklearn has no cosine KMeans → fixture = `normalize(X)` + `KMeans` euclidean; assert TS spherical k-means matches.
- Spectral cosine: sklearn has no metric param → generate via `SpectralClustering(affinity='precomputed')` fed a cosine affinity matrix.

**Effort:** M (~1.5 weeks).

---

### Phase 2 — HDBSCAN (Tier 1 #1) (effort: L)

**Items:** the full HDBSCAN estimator, built on Phase 0 primitives.

**Deliverables:**

- `condensation_tree.ts` — condensed-tree builder over the MST + Excess-of-Mass (EOM) stability selection.
- `hdbscan.ts` — `HDBSCAN` class implementing `BaseClustering<HDBSCANParams>`, emitting `labels_` (with `-1`) and `probabilities_`. Optional `exemplars_` when `storeExemplars=true`.
- `metric: 'euclidean' | 'manhattan'` natively; cosine via `metric='precomputed'` + cosine distance matrix from Phase 1.

**Test/fixture strategy:**

- `generate_hdbscan.py` mirroring `generate_spectral.py`: `make_blobs/make_moons/make_circles`, fixed `random_state`, PARAM_GRID over `min_cluster_size`, `min_samples`, `cluster_selection_epsilon`, `cluster_selection_method`. Dump `{X, params, labels, probabilities}`.
- Cosine variant: `pairwise_distances(X, metric='cosine')` → `HDBSCAN(metric='precomputed')`.
- Parity: labels up to permutation **and** consistent `-1` handling; `probabilities_` within tolerance.
- HDBSCAN is fully deterministic (no `random_state`) → tight tolerances are safe.

**Effort:** L (~3–4 weeks; condensed tree + EOM is the bulk).

---

### Phase 3 — Cluster-representation accessors (Tier 3 #8) (effort: M)

**Items:** uniform `centroids_` / `getCentroids()` (KMeans), `medoidIndices_` / `computeMedoids()` (Agglomerative, Spectral, SOM-cluster), `exemplarIndices_` (HDBSCAN), and `KMeans.predict()` (nearest centroid).

**Why this order:** This is the **enabler for tracking**. Every estimator must surface a representative vector uniformly. KMeans already has centroids; the gap is Agglomerative/Spectral/HDBSCAN. Build after HDBSCAN so HDBSCAN's exemplar accessor lands in the same pass.

**Deliverables:**

- `representations.ts` — `ClusterRepresentations` interface + `select_medoids(X, labels, nClusters, metric)`.
- `medoid_selection.ts` — O(n·k) closest-sample-to-cluster-mean finder.
- `KMeans.predict(X)` — nearest centroid via `pairwiseDistanceMatrix`. (The only predict() worth shipping; Spectral/Agglomerative predict explicitly **not provided**, documented loudly.)

**Test/fixture strategy:**

- KMeans centroids + `predict()`: extend `generate.py` to dump `cluster_centers_` and `predict()` on held-out test data; assert `rtol=1e-5` on centroids and exact label parity on predict.
- Agglomerative medoids: `generate_agglomerative_medoids.py` computes medoids post-hoc from sklearn labels; assert index identity (store distances too, to disambiguate tie-breaks across platforms).

**Effort:** M (~1.5 weeks).

---

### Phase 4 — Cross-window cluster tracking (Tier 3 #6) (effort: M)

**Items:** `trackClusters(prev, curr, options)` — the temporal feature.

**Why this order:** Depends on Phase 3 (representation vectors) and Phase 1 (cosine, the default tracking metric). Most useful once HDBSCAN (Phase 2) produces the snapshots, but works with any estimator exposing a representative vector.

**Deliverables:**

- `cluster_tracking.ts` — `trackClusters`, Hungarian (`linear_sum_assignment`-equivalent) bipartite matching on a cosine cost matrix, transition emission (`PERSIST/EMERGE/DIE/MERGE/SPLIT`), stable lifeline IDs. Stateless: caller owns the `TrackingResult` carried between frames.
- Cost-matrix pruning (cost > `1 − threshold` → ∞) to keep Hungarian tractable.

**Test/fixture strategy:**

- `generate_tracking.py` using `scipy.optimize.linear_sum_assignment` (in venv): synthesize drifting `make_blobs` snapshots, fit sklearn KMeans per snapshot, compute cosine cost matrices, dump `{prev_labels, curr_labels, prev_centroids, curr_centroids, cost_matrix, assignment, transitions}`.
- This is a **synthetic deterministic** reference (sklearn has no native tracker), not estimator parity.

**Effort:** M (~1.5–2 weeks; the Hungarian implementation in pure TS + transition heuristics are the risk).

---

### Phase 5 — Convenience & scale (optional, demand-driven)

Ship only when a consumer asks. None block TDT.

| Item                                                                                           | Trigger to build                                                                                                                  |
| ---------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------- |
| **predict() + toJSON/fromJSON** (Tier 1 #3, scoped to KMeans serialize + predict from Phase 3) | Persisted `cluster_id` tables / cross-process model reuse needed.                                                                 |
| **PCA public estimator** (Tier 3 #7)                                                           | 1536-d embeddings cluster too slowly raw and pre-projection is wanted. Low risk, fixtures trivial (sklearn PCA, sign-normalized). |
| **MiniBatchKMeans** (Tier 2 #5)                                                                | Streaming centroid updates at scale. Quality/centroid-proximity fixtures only (RNG won't match).                                  |
| **Time-decay sample weights** (Tier 2 #4)                                                      | Emphasize-recent-data semantics on KMeans/Agglomerative. Split DenStream out (XL, separate).                                      |
| **DBSCAN/OPTICS** (Tier 1 alt)                                                                 | A consumer needs fixed-`eps` semantics HDBSCAN doesn't give.                                                                      |

UMAP (XL) remains out of scope.

---

## 3. Per-Item Decision Table

| Item                         | Tier        | Effort | Feasibility     | Recommendation                            |
| ---------------------------- | ----------- | ------ | --------------- | ----------------------------------------- |
| **HDBSCAN**                  | 1 #1        | L      | moderate        | **SHIP — Phase 2 (core TDT detector)**    |
| **Cosine first-class**       | 1 #2        | M      | moderate        | **SHIP — Phase 1 (TDT metric)**           |
| **Representation accessors** | 3 #8        | M      | moderate        | **SHIP — Phase 3 (tracking enabler)**     |
| **Cross-window tracking**    | 3 #6        | M      | moderate        | **SHIP — Phase 4 (the temporal feature)** |
| **KMeans predict()**         | 1 #3 / 3 #8 | S      | easy            | **SHIP — folded into Phase 3**            |
| **PCA public estimator**     | 3 #7        | M      | moderate        | DEFER — Phase 5, low-risk                 |
| **MiniBatchKMeans**          | 2 #5        | M      | straightforward | DEFER — Phase 5, demand-driven            |
| **Sample-weight decay**      | 2 #4        | M      | moderate        | DEFER — Phase 5; split DenStream          |
| **DBSCAN / OPTICS**          | 1 alt       | M      | moderate        | DEFER — subsumed by HDBSCAN (YAGNI)       |
| **Spectral/Agglo predict()** | 1 #3        | L      | hard            | **DON'T BUILD** — no principled predict   |
| **Spectral/Agglo serialize** | 1 #3        | L      | hard            | DEFER — Phase 5, KMeans-only first        |
| **UMAP**                     | 3 #7        | XL     | hard            | **LEAVE EXTERNAL** — see §6               |

---

## 4. API Design — the highest-leverage Tier-1 bundle

Consistent with the existing `BaseClustering` interface in `src/clustering/types.ts`.

### Naming-convention note (load-bearing)

The user's global rule mandates snake_case for all identifiers. **The existing codebase is uniformly camelCase** (`BaseClustering.fitPredict`, `KMeansParams.nClusters`, etc.). The constitution forbids backwards-compat shims and aliasing. Renaming the entire public API to snake_case is a large breaking change outside TDT scope. **Decision: new TDT code matches the surrounding camelCase public API**; the snake_case rule is flagged as a separate, repo-wide refactor to raise with the user before acting. No mixed convention.

### 4.1 Cosine metric extensions

```typescript
export interface KMeansParams extends BaseClusteringParams {
  maxIter?: number;
  tol?: number;
  nInit?: number;
  /** Distance metric. 'cosine' = spherical k-means (L2-normalize then Lloyd). Default 'euclidean'. */
  metric?: 'euclidean' | 'cosine';
  /** Force L2 normalization of inputs before fitting. Implied true when metric='cosine'. */
  normalize?: boolean;
}

export interface SpectralClusteringParams extends BaseClusteringParams {
  affinity?:
    | 'rbf'
    | 'nearest_neighbors'
    | 'cosine' // NEW
    | 'precomputed'
    | ((X: DataMatrix) => tf.Tensor2D);
  // ...existing fields unchanged...
}

export type ClusteringMetric = 'euclidean' | 'manhattan' | 'cosine';
```

```typescript
/** Cosine affinity A[i,j] = 1 - cosine_distance(x_i, x_j), symmetrized, diag forced to 1. */
export function compute_cosine_affinity(points: tf.Tensor2D): tf.Tensor2D;
```

### 4.2 HDBSCAN

```typescript
export interface HDBSCANParams extends CoreClusteringParams {
  // NOTE: no nClusters
  /** Minimum cluster size; smaller groups become noise. Default 5, >= 2. */
  minClusterSize?: number;
  /** Core-distance neighborhood size. Defaults to minClusterSize. >= 1. */
  minSamples?: number;
  /** 'euclidean' | 'manhattan' native; 'precomputed' for a cosine/other distance matrix. */
  metric?: 'euclidean' | 'manhattan' | 'precomputed';
  /** Flat-cut epsilon merging clusters below this density level. Default 0. */
  clusterSelectionEpsilon?: number;
  /** Cluster extraction. Default 'eom' (Excess of Mass). */
  clusterSelectionMethod?: 'eom' | 'leaf';
  /** Store one exemplar (most-persistent point) per cluster. Default false. */
  storeExemplars?: boolean;
}

export class HDBSCAN implements BaseClustering<HDBSCANParams> {
  readonly params: HDBSCANParams;
  /** Cluster labels: integers >= 0, -1 for noise. Null until fit(). */
  labels_: number[] | null = null;
  /** Strength of each sample's membership in its cluster, in [0,1]. */
  probabilities_: number[] | null = null;
  /** Exemplar sample index per cluster id. Populated when storeExemplars=true. */
  exemplarIndices_: Map<number, number> | null = null;

  constructor(params?: Partial<HDBSCANParams>);
  /** When metric='precomputed', X is an (n,n) distance matrix; otherwise (n,features). */
  fit(X: DataMatrix): Promise<void>;
  fitPredict(X: DataMatrix): Promise<number[]>; // includes -1
  dispose(): void;
}
```

**No `predict(X)`** on HDBSCAN. `approximatePredict` against a saved condensed tree is explicitly **out of scope**. Document fit-only, like Agglomerative.

### 4.3 KMeans predict + representation accessors

```typescript
export interface ClusterRepresentations {
  centroids_?: tf.Tensor2D | null; // KMeans, HDBSCAN if computed
  medoidIndices_?: Int32Array | null; // Agglomerative, Spectral
  exemplarIndices_?: Map<number, number> | null; // HDBSCAN
}

export function select_medoids(
  X: DataMatrix,
  labels: number[],
  nClusters: number,
  metric?: ClusteringMetric,
): Promise<{ indices: Int32Array; distances: Float32Array }>;

export class KMeans
  implements BaseClustering<KMeansParams>, ClusterRepresentations
{
  centroids_: tf.Tensor2D | null = null; // already exists internally — promote to public
  inertia_: number | null = null;
  /** Assign each row of X to the nearest fitted centroid. Throws if not fitted. */
  predict(X: DataMatrix): Promise<number[]>;
  getCentroids(): number[][];
}
```

### 4.4 Hard-parts callout

1. **Minimum Spanning Tree in TF.js.** No sparse-graph ops exist. Prim's over a _dense_ (n,n) mutual-reachability matrix is O(n²) memory / O(n² log n) time — the practical scalability ceiling (~5k samples). Implement in plain JS over a `Float64Array` (not tensors), as the existing eigen/graph code already does. **Risk:** off-by-one in priority-queue update; mitigate with golden-graph unit tests.
2. **Condensed tree + EOM stability.** The single hardest piece. Builds _density-ordered_ from the MST, tracks each cluster's lifetime (`λ_birth → λ_death`), and selects by Excess-of-Mass stability. **Risks:** tree-traversal off-by-one; FP ties in degenerate equidistant cases. **Mitigate:** match sklearn 1.3.2 tolerance defaults exactly; comprehensive fixture grid.
3. **Neighbor index extraction.** The k-NN affinity builder computes neighbor indices/distances then discards them. **Root-cause fix (not a wrapper):** change its signature to return `{affinity, neighbor_indices, neighbor_distances}` and update callers. k-distance / core-distance fall straight out.
4. **The `-1` noise label invariant.** Breaks the documented repo invariant. Must be a recorded decision (Phase 0), and every validation metric must filter `-1` before pairwise distances — otherwise all-noise or single-cluster-plus-noise input divides by zero. A correctness requirement.
5. **Cosine in HDBSCAN.** sklearn HDBSCAN _rejects_ `metric='cosine'`. The cosine path goes through `metric='precomputed'` + a cosine distance matrix. The TS API must accept a precomputed (n,n) matrix in `fit` when `metric='precomputed'`.

---

## 5. Proposed Backlog.md Tasks

Atomic, independent within a phase, no forward references. Numbered from 49 (highest existing top-level id is 48). Dependencies only ever point backward.

### task-49 — Density-clustering shared utilities: MST, k-distance, mutual-reachability

Add the graph/density primitives (Prim MST, k-distance vector, mutual-reachability matrix) that HDBSCAN requires, as standalone unit-tested utilities.

- [ ] MST (Prim) over a dense distance matrix matches golden edge sets on small graphs.
- [ ] k-distance / core-distance vectors extracted from a k-NN scan.
- [ ] mutual-reachability computes `max(core_k(i), core_k(j), d(i,j))`.
- [ ] k-NN affinity builder refactored to also return `{neighbor_indices, neighbor_distances}`; callers updated (no overload/shim).
- [ ] Tensor-memory safe; pure-JS where graph traversal is involved.
      Depends on: none.

### task-50 — Decision: noise-label (-1) semantics and validation-metric handling

Record the decision that density estimators emit `-1` for noise, amend the repo invariant, and make all validation metrics filter `-1` before distance computation.

- [ ] `backlog/decisions/` entry amending the "no algorithm emits -1" invariant.
- [ ] `silhouette`, `calinski_harabasz`, `davies_bouldin` filter `-1` samples; no division-by-zero on all-noise / single-cluster-plus-noise inputs.
- [ ] Unit tests cover all-noise and one-cluster-plus-noise edge cases.
      Depends on: none.

### task-51 — Cosine as a first-class metric across estimators

Add cosine distance/affinity to KMeans (spherical), Spectral (`affinity='cosine'`), confirm Agglomerative cosine, and add an optional `metric` arg to validation metrics.

- [ ] `compute_cosine_affinity` utility added and unit-tested.
- [ ] `KMeansParams.metric='cosine'` performs spherical k-means; k-means++ and Lloyd route through `pairwiseDistanceMatrix(metric)` (no hardcoded euclidean).
- [ ] `SpectralClusteringParams.affinity='cosine'` produces a valid embedding.
- [ ] Agglomerative `metric='cosine'` validated against sklearn fixtures (label parity up to permutation).
- [ ] Validation metrics accept `metric`; all internal callers updated explicitly.
- [ ] Standalone cosine pairwise-distance fixture from `sklearn.metrics.pairwise_distances` validates the cosine path.
      Depends on: none (parallel with 49/50).

### task-52 — HDBSCAN condensed tree and EOM cluster selection

Build the condensed-tree builder over the MST and the Excess-of-Mass stability selection, as a unit-tested module independent of the estimator wrapper.

- [ ] Condensed tree builds from an MST + mutual-reachability ordering.
- [ ] EOM stability selection extracts clusters matching sklearn on golden small inputs.
- [ ] `clusterSelectionMethod: 'eom' | 'leaf'` both supported.
- [ ] Degenerate cases handled (all points equidistant, single cluster) without NaN.
      Depends on: task-49.

### task-53 — HDBSCAN estimator

Implement the `HDBSCAN` class (`labels_` with `-1`, `probabilities_`, optional exemplars) on top of the density utilities and condensed tree, with sklearn parity fixtures.

- [ ] `HDBSCAN implements BaseClustering<HDBSCANParams>`; `fit`/`fitPredict`/`dispose`.
- [ ] `metric: 'euclidean' | 'manhattan'` native; `'precomputed'` accepts an (n,n) distance matrix.
- [ ] `generate_hdbscan.py` produces fixtures over `min_cluster_size`/`min_samples`/`cluster_selection_epsilon`/`cluster_selection_method`; cosine via precomputed cosine matrix.
- [ ] Reference test: labels match sklearn up to permutation with consistent `-1`; `probabilities_` within tolerance.
- [ ] `storeExemplars=true` populates `exemplarIndices_`.
      Depends on: task-49, task-50, task-52; cosine fixture path uses task-51.

### task-54 — KMeans predict() and centroid accessor

Add nearest-centroid `predict(X)` and `getCentroids()` to KMeans, promoting `centroids_` to public, validated against sklearn `predict()`.

- [ ] `KMeans.predict(X)` assigns nearest centroid via `pairwiseDistanceMatrix`; throws if unfitted.
- [ ] `getCentroids()` returns `number[][]`; `centroids_` public.
- [ ] Fixture extends `generate.py` to dump `cluster_centers_` + sklearn `predict()` on held-out data; centroids match `rtol=1e-5`, predict labels exact.
      Depends on: none.

### task-55 — Cluster medoid/exemplar representation accessors

Add a uniform `ClusterRepresentations` interface and `select_medoids` utility so Agglomerative/Spectral expose representative vectors, and wire HDBSCAN exemplars in.

- [ ] `representations.ts` defines `ClusterRepresentations` + `select_medoids(X, labels, nClusters, metric)`.
- [ ] `medoid_selection.ts` finds the closest sample to each cluster mean (O(n·k)); handles empty/sparse label sets defensively.
- [ ] Agglomerative exposes `medoidIndices_` / `computeMedoids(X)`.
- [ ] `generate_agglomerative_medoids.py` produces post-hoc medoid fixtures; index identity asserted (with stored distances to disambiguate ties).
      Depends on: task-54 (shares the accessor surface); task-53 for HDBSCAN exemplars.

### task-56 — Cross-window cluster tracking (trackClusters)

Implement bipartite (Hungarian) matching of clusters across consecutive snapshots, emitting emerge/persist/merge/split/die transitions with stable lifeline IDs.

- [ ] `cluster_tracking.ts` exposes `trackClusters(prev, curr, options, prev_state?)`.
- [ ] Cosine cost matrix from representative vectors; pure-TS Hungarian assignment with cost pruning above `1 − threshold`.
- [ ] Emits `PERSIST/EMERGE/DIE/MERGE/SPLIT`; lifeline IDs persist when `prev_state` is threaded through.
- [ ] `generate_tracking.py` (scipy `linear_sum_assignment`) produces synthetic drifting-snapshot fixtures; TS assignment + transitions match the reference.
- [ ] Handles `prev_n != curr_n` (rectangular) without crashing; documented.
      Depends on: task-51 (cosine), task-54/55 (representation vectors).

### task-57 (Phase 5, deferred) — Public PCA estimator

Extract the internal power-iteration PCA into a standalone `PCA` estimator with `fit/transform/fitTransform/inverseTransform` + serialization, validated against sklearn PCA with sign normalization.

- [ ] `PCA` with transform/inverse/explained-variance accessors and `toJSON/fromJSON`.
- [ ] `generate_pca.py` (`svd_solver='full'`) fixtures; compare components/transform up to per-axis sign (svd_flip convention).
- [ ] `fitPredict` throws (PCA is not a clusterer); input validation rejects `nComponents > nFeatures`.
      Depends on: none. Build only on demand.

### task-58 (Phase 5, deferred) — KMeans serialization (toJSON/fromJSON)

Add JSON serialization to KMeans (centroids, params, inertia) with round-trip + predict-after-restore tests; explicitly do not provide Spectral/Agglomerative predict.

- [ ] `KMeans.toJSON()/fromJSON()` round-trips centroids and reproduces `predict()` exactly.
- [ ] Documented that Spectral/Agglomerative predict() are not provided and why.
      Depends on: task-54.

---

## 6. Risks, Open Questions, and Scope Cuts

### Risks

- **Condensed tree / EOM is the schedule risk.** The one piece with no analog in the codebase and subtle density-ordered semantics. Isolating it (task-52) with golden fixtures before the estimator (task-53) is the mitigation.
- **O(n²) memory ceiling.** MST and all distance matrices are dense; no spatial index exists. Document a practical `maxSamples ≈ 5k` for HDBSCAN and `maxClusters ≈ 100` guidance for Hungarian tracking (O(k³)). For a per-session windowed browsing timeline this is almost certainly fine — confirm window sizes.
- **Tracking transition ambiguity.** A 2→2 match can be (persist+persist) or (merge+split) depending on `threshold`. Inherent. Mitigate with documented threshold semantics and synthetic reference fixtures.
- **Naming-convention fork.** New camelCase code matches the existing API but violates the global snake_case rule. Flagged for a separate decision; do not silently mix conventions.

### Open questions (for the consumer / maintainer)

1. **Typical window size** for a browsing-timeline snapshot? If routinely >5k events, MST/affinity O(n²) will bite and we must prioritize spatial indexing (currently un-scoped).
2. **Embedding dimensionality** — if 1536-d raw, do we need PCA pre-projection (task-57) _before_ HDBSCAN for quality/speed? May promote PCA from deferred to in-scope.
3. **Is `-1` noise acceptable** to bergamot, or must noise fold into a catch-all topic? Drives the validation-metric and downstream-labeler contract (task-50).
4. **Lifeline-ID persistence ownership** — confirm bergamot will thread `TrackingResult` between frames (stateless design, per YAGNI), vs. wanting the library to own tracking state.

### YAGNI / scope cuts (decisive)

- **Cut DBSCAN/OPTICS** from the critical path — subsumed by HDBSCAN.
- **Cut UMAP entirely.** Not in venv, XL, non-deterministic. If needed, the consumer runs `umap-learn` upstream and feeds reduced vectors in.
- **Cut Spectral/Agglomerative predict() and serialization-with-affinity.** No principled predict; affinity serialization is O(n²) and OOM-prone. KMeans-only serialization in Phase 5.
- **Cut DenStream** from the sample-weight item — XL, separate future work.
- **Cut `approximatePredict` for HDBSCAN.** HDBSCAN ships fit-only.

### Constitution compliance

- **No backwards-compat shims:** the k-NN affinity refactor (task-49) changes the signature and updates callers rather than adding an overload; the validation-metric `metric` param updates all callers rather than defaulting silently behind a shim.
- **Root-cause:** neighbor indices are surfaced from where they are already computed, not recomputed in a wrapper.
- **No unsafe casts:** all new APIs use concrete types (`number[][]`, `Int32Array`, `Map<number, number>`, tagged unions); no `as any`/`as unknown`/`as never`. `labels_`/`centroids_` are nullable only because fitted-state is genuinely absent before `fit()`.

---

## Appendix — Fixture availability (verified against the venv: sklearn 1.3.2, numpy 1.24.4, scipy 1.10.1)

| Algorithm       | Source                                                | In venv | Notes                                                                                                                             |
| --------------- | ----------------------------------------------------- | ------- | --------------------------------------------------------------------------------------------------------------------------------- |
| HDBSCAN         | `sklearn.cluster.HDBSCAN`                             | ✅      | Fully deterministic (no `random_state`). Emits `-1`. **Rejects `metric='cosine'`** → use `metric='precomputed'` + cosine matrix.  |
| DBSCAN          | `sklearn.cluster.DBSCAN`                              | ✅      | Deterministic given fixed row order. `metric='cosine'` works directly. Emits `-1`.                                                |
| OPTICS          | `sklearn.cluster.OPTICS`                              | ✅      | Deterministic. `metric='cosine'` works directly. Emits `-1`.                                                                      |
| PCA             | `sklearn.decomposition.PCA`                           | ✅      | Sign-ambiguous per component — compare with svd_flip normalization; use `svd_solver='full'`.                                      |
| MiniBatchKMeans | `sklearn.cluster.MiniBatchKMeans`                     | ✅      | Deterministic only with fixed `random_state`; TS RNG won't match → quality/centroid-proximity comparison, not exact-label parity. |
| cosine pairwise | `sklearn.metrics.pairwise_distances(metric='cosine')` | ✅      | Deterministic. Validates the cosine distance path independently.                                                                  |
| UMAP            | external `umap-learn` (module `umap`)                 | ❌      | Not installed; stochastic (SGD + numba); cannot be value-asserted across platforms. **Leave external.**                           |
