# Topic Detection and Tracking (TDT) Upgrade Plan for clustering-js

**Audience:** the bergamot browsing-timeline consumer and the clustering-js maintainers.
**Status:** decisive implementation plan, decomposed into parent `task-49` with subtasks `task-49.1`–`task-49.10`.

---

## 1. Executive Summary

Topic Detection and Tracking over a browsing timeline is, at its core, two distinct problems:

1. **Topic detection** — given a window of embedded browsing events, find coherent topics _without knowing how many there are_, and flag noise (incidental, one-off pages). This is a density-clustering problem on embedding vectors, where the natural metric is cosine.
2. **Topic tracking** — given clusters from consecutive windows, decide which topic at window _t_ _is_ which topic at window _t−1_, and emit `emerge / persist / merge / split / die` transitions with stable lifeline identities. This is the _only genuinely temporal_ feature in the entire backlog.

Everything else (PCA, MiniBatchKMeans, sample weights, predict/serialize, representation accessors, UMAP) is supporting infrastructure or convenience.

### The minimum viable bundle for TDT

| Capability                                                         | Why it is non-negotiable for TDT                                                                                                                                         |
| ------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **HDBSCAN**                                                        | The field-default topic detector (BERTopic's default). Discovers variable-density topics, no cluster count required, emits noise for incidental pages.                   |
| **Cosine as a first-class metric**                                 | Browsing-event embeddings are direction-meaningful, magnitude-noisy. Cosine is the correct topic-similarity metric; Euclidean on raw embeddings clusters poorly.         |
| **Cross-window cluster tracking**                                  | The actual "Tracking" in Topic Detection and **Tracking**. Without it you have repeated snapshots, not a timeline.                                                       |
| **Cluster-representation accessors** (centroids/medoids/exemplars) | Tracking matches topics by their representative vectors. Tracking is dead-on-arrival without a uniform way to get a topic's representative point out of every estimator. |

That is the bundle. **DBSCAN, predict()+serialize, PCA, MiniBatchKMeans, and sample-weights are each a deferrable optimization or convenience, not a TDT enabler.** UMAP stays external (see §6).

### The single genuinely-temporal feature

**`track_clusters(prev, curr, options, prev_state?)`** is the only item whose semantics are inherently about _time_. It is the second-highest priority after HDBSCAN. It has a hard practical dependency on representation accessors and is most useful once HDBSCAN exists as the snapshot producer.

### Opinionated cuts

- **Defer DBSCAN/OPTICS.** Once HDBSCAN is committed, a fixed-`eps` stepping stone is wasted motion (YAGNI). HDBSCAN's k-distance / mutual-reachability machinery _subsumes_ DBSCAN. Add DBSCAN later only if a consumer explicitly needs fixed-`eps` semantics.
- **Leave UMAP external.** Non-deterministic, XL effort, not in the venv, fixtures can't be asserted by value.
- **No `predict()` for Spectral or Agglomerative.** There is no principled predict for these without full recompute.

---

## 2. Sequenced Roadmap

Ordering is driven by three hard dependencies:

1. Tracking matches by centroid/medoid/exemplar, so it needs representation accessors.
2. HDBSCAN's exemplar accessor is built alongside HDBSCAN, but the _generic_ centroid/medoid accessors are independent and unblock tracking for KMeans/Agglomerative/Spectral immediately.
3. Cosine support is independent of the above and is a prerequisite for _good_ topic clustering and _good_ tracking.

### Phase 0 — Foundations & shared utilities (`task-49.1`, `task-49.2`) (effort: M)

HDBSCAN depends on graph/density machinery the rest of the library does not need yet (minimum spanning tree, k-distance, mutual-reachability, condensed tree). Building and unit-testing these primitives in isolation de-risks the large estimator. The `-1` noise-label contract is a cross-cutting design choice settled before any density estimator or validation metric touches it.

**Deliverables:**

- `src/distance/kdistance.ts` — k-distance vector + core-distance extraction from a (block-wise) k-NN scan. `compute_knn_affinity` in `src/graph/affinity.ts` also returns `{ neighbor_indices, neighbor_distances }`, surfacing the indices and distances the scan already computes.
- `src/graph/mutual_reachability.ts` — `d_mreach(i, j) = max(core_k(i), core_k(j), d(i, j))`.
- `src/graph/minimum_spanning_tree.ts` — Prim's algorithm over a dense distance / mutual-reachability matrix.
- A decision record (`backlog/decisions/`) on noise labels: **non-density estimators emit dense `0..n_clusters-1`; density estimators emit `-1` for noise.** Validation metrics filter `-1` before computing pairwise distances.

**Test/fixture strategy:** pure unit tests for MST (known small graphs, golden edge sets), k-distance, and mutual-reachability — deterministic numeric primitives, no sklearn needed. A standalone cosine pairwise-distance fixture from `sklearn.metrics.pairwise_distances(metric='cosine')` validates the cosine path independently.

---

### Phase 1 — Cosine as a first-class metric (`task-49.3`) (effort: M)

Cosine metric/affinity across Agglomerative (`metric='cosine'`), Spectral (`affinity='cosine'`), KMeans (`metric='cosine'` via L2-normalize = spherical k-means), and an optional `metric` parameter on the validation metrics. Independent of Phase 0; runs in parallel if staffed. Lands before HDBSCAN so HDBSCAN's cosine path (via precomputed) reuses a tested cosine matrix builder.

**Deliverables:**

- `compute_cosine_affinity` in `src/graph/affinity.ts`, reachable through the `compute_affinity_matrix` dispatcher.
- KMeans `metric: 'euclidean' | 'cosine'`. Cosine L2-normalizes then runs k-means++ seeding and Lloyd assignment through `pairwise_distance_matrix(points, metric)`.
- Spectral `affinity='cosine'` dispatching to `compute_cosine_affinity`.
- Validation metrics (`silhouette`, `davies_bouldin`) gain an optional `metric` parameter; `calinski_harabasz` is variance-based and metric-independent. Every internal caller passes `metric` explicitly.

**Test/fixture strategy (sklearn 1.3.2, already in venv):**

- Agglomerative `metric='cosine'`: direct sklearn fixtures, exact-up-to-permutation label parity.
- KMeans cosine: sklearn has no cosine KMeans → fixture = `normalize(X)` + Euclidean `KMeans`; assert spherical k-means matches.
- Spectral cosine: sklearn has no metric param → generate via `SpectralClustering(affinity='precomputed')` fed a cosine affinity matrix.

---

### Phase 2 — HDBSCAN (`task-49.4`, `task-49.5`) (effort: L)

The full HDBSCAN estimator, built on Phase 0 primitives.

**Deliverables:**

- `src/graph/condensation_tree.ts` — condensed-tree builder over the MST + Excess-of-Mass (EOM) stability selection (`task-49.4`).
- `src/clustering/hdbscan.ts` — `HDBSCAN` implementing `BaseClustering<HDBSCANParams>`, emitting `labels_` (with `-1`) and `probabilities_`, plus `exemplar_indices_` when `store_exemplars=true` (`task-49.5`).
- `metric: 'euclidean' | 'manhattan'` natively; cosine via `metric='precomputed'` + a cosine distance matrix from Phase 1.

**Test/fixture strategy:** `generate_hdbscan.py` mirroring `generate_spectral.py`: `make_blobs/make_moons/make_circles`, fixed `random_state`, a grid over `min_cluster_size`, `min_samples`, `cluster_selection_epsilon`, `cluster_selection_method`. Cosine variant via `pairwise_distances(X, metric='cosine')` → `HDBSCAN(metric='precomputed')`. Labels match up to permutation with consistent `-1`; `probabilities_` within tolerance. HDBSCAN is fully deterministic, so tight tolerances are safe.

---

### Phase 3 — Cluster-representation accessors (`task-49.6`, `task-49.7`) (effort: M)

The enabler for tracking: every estimator surfaces a representative vector uniformly. KMeans already computes centroids; the gap is Agglomerative and Spectral, which produce only labels.

**Deliverables:**

- `src/clustering/kmeans.ts` — `predict(X)` (nearest centroid via `pairwise_distance_matrix`) and `get_centroids()` (`task-49.6`).
- `src/clustering/representations.ts` — `ClusterRepresentations` interface (`centroids_`, `medoid_indices_`, `exemplar_indices_`).
- `src/clustering/medoid_selection.ts` — `select_medoids(X, labels, n_clusters, metric)`, an O(n·k) closest-sample-to-cluster-mean finder. Agglomerative and Spectral expose `medoid_indices_` / `compute_medoids(X)` (`task-49.7`).

HDBSCAN owns its own `exemplar_indices_` through `store_exemplars` (`task-49.5`); the `ClusterRepresentations` contract carries an optional `exemplar_indices_` field so density and partition estimators report representatives through one shape.

**Test/fixture strategy:** KMeans centroids + `predict()` extend `generate.py` to dump `cluster_centers_` and `predict()` on held-out data; assert `rtol=1e-5` on centroids and exact label parity on predict. `generate_agglomerative_medoids.py` computes medoids post-hoc from sklearn labels; assert index identity (store distances too, to disambiguate tie-breaks).

---

### Phase 4 — Cross-window cluster tracking (`task-49.8`) (effort: M)

`track_clusters(prev, curr, options, prev_state?)` — the temporal feature. Depends on representation vectors (Phase 3) and cosine (Phase 1). Most useful once HDBSCAN produces the snapshots, but works with any estimator exposing a representative vector.

**Deliverables:**

- `src/clustering/cluster_tracking.ts` — `track_clusters`, a pure-TypeScript Hungarian (`linear_sum_assignment`-equivalent) bipartite matching on a cosine cost matrix, transition emission (`PERSIST/EMERGE/DIE/MERGE/SPLIT`), stable lifeline IDs. Stateless: the caller owns the `TrackingResult` carried between frames.
- Cost-matrix pruning (cost > `1 − threshold` → ∞) to keep Hungarian tractable.

**Test/fixture strategy:** `generate_tracking.py` using `scipy.optimize.linear_sum_assignment`: synthesize drifting `make_blobs` snapshots, fit sklearn KMeans per snapshot, compute cosine cost matrices, dump `{prev_labels, curr_labels, prev_centroids, curr_centroids, cost_matrix, assignment, transitions}`. This is a synthetic deterministic reference (sklearn has no native tracker), not estimator parity.

---

### Phase 5 — Convenience & scale (`task-49.9`, `task-49.10`; demand-driven)

Ship when a consumer asks. None block TDT.

| Item                                                         | Trigger to build                                                                                   |
| ------------------------------------------------------------ | -------------------------------------------------------------------------------------------------- |
| **Public PCA estimator** (`task-49.9`, `src/decomposition/`) | 1536-d embeddings cluster too slowly raw and pre-projection is wanted. Low risk, fixtures trivial. |
| **KMeans `to_json`/`from_json`** (`task-49.10`)              | Persisted `cluster_id` tables / cross-process model reuse needed.                                  |
| **MiniBatchKMeans**                                          | Streaming centroid updates at scale. Quality/centroid-proximity fixtures only (RNG won't match).   |
| **Time-decay sample weights**                                | Emphasize-recent-data semantics on KMeans/Agglomerative. DenStream is separate XL work.            |
| **DBSCAN/OPTICS**                                            | A consumer needs fixed-`eps` semantics HDBSCAN doesn't give.                                       |

UMAP (XL) remains out of scope.

---

## 3. Per-Item Decision Table

| Item                           | Effort | Feasibility     | Recommendation                          |
| ------------------------------ | ------ | --------------- | --------------------------------------- |
| **HDBSCAN**                    | L      | moderate        | **SHIP — `task-49.4`/`task-49.5`**      |
| **Cosine first-class**         | M      | moderate        | **SHIP — `task-49.3`**                  |
| **Representation accessors**   | M      | moderate        | **SHIP — `task-49.6`/`task-49.7`**      |
| **Cross-window tracking**      | M      | moderate        | **SHIP — `task-49.8`**                  |
| **KMeans `predict()`**         | S      | easy            | **SHIP — `task-49.6`**                  |
| **Public PCA estimator**       | M      | moderate        | DEFER — `task-49.9`, low-risk           |
| **KMeans serialization**       | S      | easy            | DEFER — `task-49.10`, demand-driven     |
| **MiniBatchKMeans**            | M      | straightforward | DEFER — Phase 5, demand-driven          |
| **Sample-weight decay**        | M      | moderate        | DEFER — Phase 5; DenStream separate     |
| **DBSCAN / OPTICS**            | M      | moderate        | DEFER — subsumed by HDBSCAN (YAGNI)     |
| **Spectral/Agglo `predict()`** | L      | hard            | **DON'T BUILD** — no principled predict |
| **UMAP**                       | XL     | hard            | **LEAVE EXTERNAL** — see §6             |

---

## 4. API Design — the highest-leverage bundle

Consistent with the `BaseClustering` interface in `src/clustering/types.ts`. Identifiers follow the library convention: `snake_case` for functions/methods/properties/params, `PascalCase` for classes/interfaces/type aliases, fitted attributes use sklearn-style trailing underscores.

### 4.1 Cosine metric extensions

```typescript
export interface KMeansParams extends BaseClusteringParams {
  max_iter?: number;
  tol?: number;
  n_init?: number;
  /** Distance metric. 'cosine' = spherical k-means (L2-normalize then Lloyd). Default 'euclidean'. */
  metric?: 'euclidean' | 'cosine';
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

/** Cosine affinity A[i,j] = 1 - cosine_distance(x_i, x_j), symmetrized, diag forced to 1. */
export function compute_cosine_affinity(points: tf.Tensor2D): tf.Tensor2D;
```

### 4.2 HDBSCAN

```typescript
export interface HDBSCANParams extends CoreClusteringParams {
  // No n_clusters.
  /** Minimum cluster size; smaller groups become noise. Default 5, >= 2. */
  min_cluster_size?: number;
  /** Core-distance neighborhood size. Defaults to min_cluster_size. >= 1. */
  min_samples?: number;
  /** 'euclidean' | 'manhattan' native; 'precomputed' for a cosine/other distance matrix. */
  metric?: 'euclidean' | 'manhattan' | 'precomputed';
  /** Flat-cut epsilon merging clusters below this density level. Default 0. */
  cluster_selection_epsilon?: number;
  /** Cluster extraction. Default 'eom' (Excess of Mass). */
  cluster_selection_method?: 'eom' | 'leaf';
  /** Store one exemplar (most-persistent point) per cluster. Default false. */
  store_exemplars?: boolean;
}

export class HDBSCAN implements BaseClustering<HDBSCANParams> {
  readonly params: HDBSCANParams;
  /** Cluster labels: integers >= 0, -1 for noise. Null until fit(). */
  labels_: number[] | null = null;
  /** Strength of each sample's membership in its cluster, in [0,1]. */
  probabilities_: number[] | null = null;
  /** Exemplar sample index per cluster id. Populated when store_exemplars=true. */
  exemplar_indices_: Map<number, number> | null = null;

  constructor(params?: Partial<HDBSCANParams>);
  /** When metric='precomputed', X is an (n,n) distance matrix; otherwise (n,features). */
  fit(X: DataMatrix): Promise<void>;
  fit_predict(X: DataMatrix): Promise<number[]>; // includes -1
  dispose(): void;
}
```

HDBSCAN is fit-only, like Agglomerative — there is no `predict(X)`.

### 4.3 KMeans predict + representation accessors

```typescript
export interface ClusterRepresentations {
  centroids_?: tf.Tensor2D | null; // KMeans
  medoid_indices_?: Int32Array | null; // Agglomerative, Spectral
  exemplar_indices_?: Map<number, number> | null; // HDBSCAN
}

export function select_medoids(
  X: DataMatrix,
  labels: number[],
  n_clusters: number,
  metric?: ClusteringMetric,
): Promise<{ indices: Int32Array; distances: Float32Array }>;

export class KMeans
  implements BaseClustering<KMeansParams>, ClusterRepresentations
{
  centroids_: tf.Tensor2D | null = null;
  inertia_: number | null = null;
  /** Assign each row of X to the nearest fitted centroid. Throws if not fitted. */
  predict(X: DataMatrix): Promise<number[]>;
  get_centroids(): number[][];
}
```

### 4.4 Hard-parts callout

1. **Minimum Spanning Tree in TF.js.** No sparse-graph ops exist. Prim's over a _dense_ (n,n) mutual-reachability matrix is O(n²) memory / O(n² log n) time — the practical scalability ceiling (~5k samples). Implement in plain JS over a `Float64Array` (not tensors), as the existing eigen/graph code does. **Risk:** off-by-one in priority-queue update; mitigate with golden-graph unit tests.
2. **Condensed tree + EOM stability.** The single hardest piece. Builds _density-ordered_ from the MST, tracks each cluster's lifetime (`λ_birth → λ_death`), and selects by Excess-of-Mass stability. **Risks:** tree-traversal off-by-one; FP ties in degenerate equidistant cases. **Mitigate:** match sklearn 1.3.2 tolerance defaults; comprehensive fixture grid.
3. **Neighbor index extraction.** `compute_knn_affinity` runs a single k-NN scan that produces neighbor indices and distances. It returns `{ affinity, neighbor_indices, neighbor_distances }`, and the k-distance / core-distance primitives consume those directly. Callers consume the single return shape.
4. **The `-1` noise label.** Density estimators emit `-1` for noise. Every validation metric filters `-1` before pairwise distances — otherwise all-noise or single-cluster-plus-noise input divides by zero. A correctness requirement, recorded as a decision in Phase 0.
5. **Cosine in HDBSCAN.** sklearn HDBSCAN rejects `metric='cosine'`. The cosine path goes through `metric='precomputed'` + a cosine distance matrix. `fit` accepts a precomputed (n,n) matrix when `metric='precomputed'`.

---

## 5. Backlog.md Tasks

Parent `task-49` is the umbrella; the work is decomposed into subtasks `task-49.1`–`task-49.10`, atomic and independent within a phase, with dependencies pointing only backward.

### `task-49.1` — Density-clustering graph primitives: minimum spanning tree, k-distance, mutual reachability

Add the graph/density primitives (Prim MST, k-distance vector, mutual-reachability matrix) that HDBSCAN requires, as standalone unit-tested utilities. `compute_knn_affinity` returns `{ neighbor_indices, neighbor_distances }` alongside the affinity matrix, and callers consume the single return shape. — _Depends on: none._

### `task-49.2` — Define noise-label (-1) semantics and make validation metrics noise-aware

Record the cluster-label contract (non-density emit dense `0..n_clusters-1`; density emit `-1`) as a decision, and make `silhouette`, `calinski_harabasz`, and `davies_bouldin` filter `-1` before distance computation, with no division-by-zero on all-noise or single-cluster-plus-noise inputs. — _Depends on: none._

### `task-49.3` — Cosine as a first-class metric across estimators and validation metrics

Add cosine affinity to KMeans (spherical), Spectral (`affinity='cosine'`), Agglomerative (`metric='cosine'`), and an optional `metric` arg to `silhouette`/`davies_bouldin`. Distance/affinity routes through `pairwise_distance_matrix` and `compute_cosine_affinity`. — _Depends on: none._

### `task-49.4` — HDBSCAN condensed tree and excess-of-mass cluster selection

Build `src/graph/condensation_tree.ts` (condensed-tree builder over the MST + EOM stability selection) as a unit-tested module independent of the estimator, matching sklearn on golden inputs. — _Depends on: `task-49.1`._

### `task-49.5` — HDBSCAN density-based clustering estimator

Implement `HDBSCAN` (`labels_` with `-1`, `probabilities_`, optional `exemplar_indices_`) on the density utilities and condensed tree, with sklearn parity fixtures. — _Depends on: `task-49.1`, `task-49.2`, `task-49.3`, `task-49.4`._

### `task-49.6` — KMeans nearest-centroid predict and centroid accessor

Add `predict(X)` (nearest centroid via `pairwise_distance_matrix`) and `get_centroids()` to KMeans, validated against sklearn `predict()`. — _Depends on: none._

### `task-49.7` — Expose cluster representative vectors via uniform medoid accessors

Add `ClusterRepresentations` and `select_medoids` so Agglomerative and Spectral expose representative vectors through the same shape as KMeans centroids. — _Depends on: `task-49.6`._

### `task-49.8` — Cross-snapshot cluster tracking via `track_clusters`

Implement bipartite (Hungarian) matching of clusters across consecutive snapshots, emitting `PERSIST/EMERGE/DIE/MERGE/SPLIT` transitions with stable lifeline IDs. Stateless. — _Depends on: `task-49.3`, `task-49.6`, `task-49.7`._

### `task-49.9` — Public PCA estimator in `src/decomposition` (Phase 5, deferred)

Provide a standalone `PCA` (`fit`/`transform`/`fit_transform`/`inverse_transform` + `to_json`/`from_json`) in `src/decomposition/pca.ts`, validated against sklearn PCA with `svd_flip` sign normalization. SOM `'linear'`/`'pca'` initialization sources its principal components from this shared computation. — _Depends on: none._

### `task-49.10` — KMeans serialization (`to_json`/`from_json`) (Phase 5, deferred)

Add JSON serialization to KMeans (centroids, params, inertia) with round-trip + predict-after-restore tests. SpectralClustering and AgglomerativeClustering are transductive and expose no predict or serialization; `docs/API.md` documents the asymmetry. — _Depends on: `task-49.6`._

---

## 6. Risks, Open Questions, and Scope Cuts

### Risks

- **Condensed tree / EOM is the schedule risk.** The one piece with no analog in the codebase and subtle density-ordered semantics. Isolating it (`task-49.4`) with golden fixtures before the estimator (`task-49.5`) is the mitigation.
- **O(n²) memory ceiling.** MST and all distance matrices are dense; no spatial index exists. Document a practical `max_samples ≈ 5k` for HDBSCAN and `max_clusters ≈ 100` guidance for Hungarian tracking (O(k³)). For a per-session windowed browsing timeline this is almost certainly fine — confirm window sizes.
- **Tracking transition ambiguity.** A 2→2 match can be (persist+persist) or (merge+split) depending on `threshold`. Inherent. Mitigate with documented threshold semantics and synthetic reference fixtures.

### Open questions (for the consumer / maintainer)

1. **Typical window size** for a browsing-timeline snapshot? If routinely >5k events, the O(n²) MST/affinity will bite and spatial indexing (currently un-scoped) must be prioritized.
2. **Embedding dimensionality** — if 1536-d raw, is PCA pre-projection (`task-49.9`) wanted _before_ HDBSCAN for quality/speed? May promote PCA from deferred to in-scope.
3. **Is `-1` noise acceptable** to bergamot, or must noise fold into a catch-all topic? Drives the validation-metric and downstream-labeler contract (`task-49.2`).
4. **Lifeline-ID persistence ownership** — confirm bergamot threads `TrackingResult` between frames (stateless design, per YAGNI), versus the library owning tracking state.

### YAGNI / scope cuts (decisive)

- **Cut DBSCAN/OPTICS** from the critical path — subsumed by HDBSCAN.
- **Cut UMAP entirely.** Not in venv, XL, non-deterministic. If needed, the consumer runs `umap-learn` upstream and feeds reduced vectors in.
- **Cut Spectral/Agglomerative `predict()` and affinity serialization.** No principled predict; affinity serialization is O(n²) and OOM-prone. KMeans-only serialization in Phase 5.
- **Cut DenStream** from the sample-weight item — XL, separate future work.
- **Cut `approximate_predict` for HDBSCAN.** HDBSCAN ships fit-only.

### Constitution compliance

- **No backwards-compat shims:** `compute_knn_affinity` changes its return shape and callers are updated rather than adding an overload; the validation-metric `metric` param updates all callers rather than defaulting silently behind a shim.
- **Root-cause:** neighbor indices are surfaced from where the k-NN scan already computes them, not recomputed in a wrapper.
- **No unsafe casts:** all APIs use concrete types (`number[][]`, `Int32Array`, `Map<number, number>`, tagged unions). `labels_`/`centroids_` are nullable only because fitted-state is genuinely absent before `fit()`.

---

## Appendix — Fixture availability (sklearn 1.3.2, numpy 1.24.4, scipy 1.10.1 in the venv)

| Algorithm       | Source                                                | In venv | Notes                                                                                                                             |
| --------------- | ----------------------------------------------------- | ------- | --------------------------------------------------------------------------------------------------------------------------------- |
| HDBSCAN         | `sklearn.cluster.HDBSCAN`                             | ✅      | Fully deterministic (no `random_state`). Emits `-1`. Rejects `metric='cosine'` → use `metric='precomputed'` + cosine matrix.      |
| DBSCAN          | `sklearn.cluster.DBSCAN`                              | ✅      | Deterministic given fixed row order. `metric='cosine'` works directly. Emits `-1`.                                                |
| OPTICS          | `sklearn.cluster.OPTICS`                              | ✅      | Deterministic. `metric='cosine'` works directly. Emits `-1`.                                                                      |
| PCA             | `sklearn.decomposition.PCA`                           | ✅      | Sign-ambiguous per component — compare with `svd_flip` normalization; use `svd_solver='full'`.                                    |
| MiniBatchKMeans | `sklearn.cluster.MiniBatchKMeans`                     | ✅      | Deterministic only with fixed `random_state`; TS RNG won't match → quality/centroid-proximity comparison, not exact-label parity. |
| cosine pairwise | `sklearn.metrics.pairwise_distances(metric='cosine')` | ✅      | Deterministic. Validates the cosine distance path independently.                                                                  |
| UMAP            | external `umap-learn` (module `umap`)                 | ❌      | Not installed; stochastic (SGD + numba); cannot be value-asserted across platforms. Leave external.                               |
