---
id: TASK-40
title: Optimize agglomerative clustering from O(n^3) to O(n^2 log n)
status: In Progress
assignee: []
created_date: '2026-03-20'
updated_date: '2026-06-08 14:30'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->

The agglomerative clustering uses a brute-force O(n^3) nearest-pair search with Array.splice() for distance matrix contraction adding O(n^2) per merge step. This makes it impractical for datasets beyond ~500 samples. The implementation should use a min-heap/priority queue for nearest-pair tracking, and for single/complete/average/ward linkage specifically, the NNCHAIN algorithm can achieve O(n^2) amortized complexity. The distance matrix contraction should use index-based active tracking instead of physically removing rows/columns.

<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->

- [x] #1 Agglomerative clustering uses priority queue or NNCHAIN for merge step
- [x] #2 Distance matrix uses index-based active tracking instead of Array.splice
- [x] #3 1000-sample dataset completes in under 5 seconds
- [x] #4 Existing reference tests still pass
- [x] #5 Benchmark added for 1000 and 5000 sample datasets

## Current State Analysis

`AgglomerativeClustering` (`src/clustering/agglomerative.ts`) wraps the merge engine `stored_nn_cluster` (`src/clustering/linkage.ts:116`). The task description is partly stale: it claims a "brute-force O(n^3) nearest-pair search with `Array.splice()`," but that implementation no longer exists. The current code already satisfies two of the five acceptance criteria:

- **Index-based active tracking** is in place: a `Uint8Array active` (`linkage.ts:122`) flags live clusters; rows/columns are never spliced.
- **A "priority queue"** exists as the stored-nearest-neighbor (Anderberg) cache: `nn`/`nn_dist` arrays (`linkage.ts:129-130`) hold each cluster's cached nearest neighbor, and the global-minimum merge pair is found by an O(n) scan of `nn_dist` (`linkage.ts:144-150`) rather than an O(n^2) scan of the full matrix.

**The real, unmet gap is the O(n^3) worst case.** Each merge runs an O(n) Lance–Williams row update (`linkage.ts:174-198`). Inside that loop, when a cluster `k`'s cached neighbor was a merge participant (`nn[k] === removed || nn[k] === survivor`, `linkage.ts:190`), `k` requires a full O(n) rescan (`linkage.ts:192`). On hub/chain-structured data, O(n) distinct clusters point at the merging pair every step, so a single merge does O(n^2) work — O(n^3) over the run. This is **not** a redundant-rescan bug fixable by dedup: each destroyed-NN cluster legitimately needs one O(n) rescan because, for complete/average/ward, the merged distance can grow and the new nearest neighbor may be any cluster. The cubic term is structural to the stored-NN design.

**Memory/time facts (dense n×n Float64 matrix):**

- `D` is `8·n^2` bytes: ~8 MB at n=1000, ~200 MB at n=5000, ~800 MB at n=10000.
- Construction materializes the full matrix: `pairwise_distance_matrix` → `.data()` (a float32 typed array, `agglomerative.ts:122`) → element-by-element copy into a second `Float64Array` (`agglomerative.ts:125-128`). The float32 staging buffer (~4·n^2) and `D` (~8·n^2) coexist, giving ~1.2 GB peak at n=10000.
- Typical run is O(n^2) time; worst case O(n^3).

**Downstream contract (verified by grep):** `labels_` (`agglomerative.ts:145-174`) is derived order-independently via Union-Find replay of all merges. `children_` (`agglomerative.ts:139-143`) is built in raw emission order and depends on merges being emitted in non-decreasing distance order. **No consumer in `src/` or `tools/` reads `children_`, `n_leaves_`, `MergeRecord.distance`, or `new_size`** — `children_` is written but never consumed. The only consumers of `AgglomerativeClustering` read `labels_` (SOM.cluster at `som.ts`, `find_optimal_clusters.ts`) or call `fit_predict`. The reference test (`agglomerative_reference.test.ts:64`) asserts only permutation-invariant label equivalence, never `children_` or distances.

**Caveat — in-progress feature in the working tree.** An unfinished `distance_threshold` / `distances_` / `'precomputed'`-metric feature is mid-flight in `agglomerative.ts` (currently producing TypeScript errors: a `'precomputed'` member added to the metric union that downstream signatures don't accept, and an optional `n_clusters` that is now possibly-`undefined`). This plan was written against the simpler API. Its cut logic, `children_`/`distances_` output, and the precomputed path all depend on that feature's final API shape — finalize it before implementing (see _Effort, Difficulty & Sequencing_).

## Strategy & Algorithm Choice

**Chosen backbone: NN-chain (reciprocal-nearest-neighbor) over the existing dense flat `Float64Array`, run to completion, then cut.** This is the only design that delivers a _guaranteed_ O(n^2) worst-case bound, eliminating the `linkage.ts:190-192` cubic rescan storm entirely (NN-chain maintains no global per-cluster NN cache, so a merge invalidates at most the chain top, rescanned once). A heap-only variant is rejected: it accelerates global-min _selection_ (O(n)→O(log n)) but leaves the destroyed-NN rebuild work untouched, so the worst case stays O(n^3) — it cannot meet the spirit of the task ("from O(n^3)").

**Reducibility — why NN-chain is exact here.** A linkage is reducible iff merging two clusters never makes the merged cluster strictly closer to any third cluster than the closer of the two originals already was. Reducibility guarantees (i) a reciprocal-NN pair is always a valid next merge, and (ii) the chain prefix stays valid after a merge (the O(n^2) bound). All four supported linkages — **single, complete, average, ward** — are Lance–Williams reducible, so NN-chain reproduces the exact agglomeration. centroid/median are _not_ reducible and are not supported (`VALID_LINKAGES`, `agglomerative.ts:47-52`); a code comment guards any future addition from routing through NN-chain.

**Full-tree-then-cut contract change (mandatory).** NN-chain emits merges in _discovery_ order, which is **not** distance-sorted. Consequently the current early-stop (`linkage.ts:138-140`, `target_merges = n - n_clusters`) is invalid — stopping early would merge an arbitrary subset. The new contract is:

1. Run NN-chain to completion: always exactly `n - 1` merges (the `n_clusters` parameter is removed from the merge engine).
2. Stable-sort the `n - 1` merges by `distance` ascending. V8's `Array.prototype.sort` is stable; use a numeric comparator. This sort is **non-negotiable even though `children_` is unused**, because the label cut depends on it.
3. Cut at `n_clusters` in `agglomerative.ts`.

**How label parity with sklearn is preserved.** The reference test asserts permutation-invariant partition equivalence, not exact integer labels. To produce the k-partition, apply only the **first `n - k` merges from the distance-sorted list** through the existing order-independent Union-Find (`agglomerative.ts:145-174`), then number connected components by first-seen root. Applying the `n - k` _smallest-distance_ merges is exactly the dendrogram cut at `n_clusters` and is equivalent to sklearn's `_hc_cut` _for the partition_. This reuses the existing Union-Find machinery and is sufficient for parity; a full `_hc_cut` max-heap reimplementation (for exact integer labels) is **deliberately not done** — no test asserts exact integers, and it would be YAGNI surplus.

**How `children_` is handled.** `children_` has zero consumers. Rather than raise a dead field to a bit-for-bit scipy contract (rejected as surplus per the constitution), build it correctly and cheaply from the same sorted list: walk the distance-sorted merges through a small relabel Union-Find (`parent[0..n-1]`, `next_node = n`, path-halving find), pushing `[min(root_a, root_b), max(...)]` and minting node id `n + k` per merge. This is the standard scipy `label()` relabel, produces sklearn-shaped `children_` (node ids `n..2n-2` in distance order), and costs only the marginal Union-Find pass on top of the sort the labels already require. No new fixtures or exact-equality assertions are added for it.

## Refactoring Required

Per **NO BACKWARDS COMPATIBILITY**: rename and re-signature in place, update all callers, add no shims, no alias re-exports, no `n_clusters`-accepting wrapper.

**`src/clustering/linkage.ts`:**

- Replace `stored_nn_cluster(D, n, n_clusters, linkage)` with `nn_chain_cluster(D, n, linkage): MergeRecord[]`. **Drop the `n_clusters` parameter** (NN-chain always runs `n - 1` merges).
- Delete the stored-NN-only machinery: the `nn`/`nn_dist` caches (`129-130`), the init loop (`132-135`), the global-min scan (`142-150`), and the incremental NN-cache maintenance branch (`189-197`). Keep `lance_williams` (`46-71`) **verbatim** and `MergeRecord` (`31-40`) unchanged.
- Add an `Int32Array chain` stack (capacity n) + `chain_len`. Implement the reciprocal-NN loop with scipy tie rules (below).
- Preserve invariants: `cluster_a = min(a, b)`, `cluster_b = max(a, b)`; capture `ni`/`nj` _before_ the Lance–Williams loop and update `size[survivor]` _after_; deactivate `removed` before the loop. The merged cluster lives under the survivor _slot_ id, which must remain the find-able representative used by the relabel Union-Find.
- Rewrite the file header (`1-24`) to describe NN-chain and its guaranteed O(n^2) bound in present tense (no "previously/worst-case-O(n^3)" framing, per documentation-style).

**Distance layout decision: keep the dense flat n×n `Float64Array`.** Condensed upper-triangular storage halves the _resident_ matrix but only moves the OOM peak out ~33% (the transient float32 `.data()` buffer coexists with the condensed array during the fill), and it introduces a strided-gather cache regression in the hot NN scan that threatens the 5s@1000 ceiling. The real prize is killing the O(n^3); condensed storage is a separable, benchmark-gated follow-up, deferred under YAGNI. NN-chain on dense storage proves guaranteed O(n^2) and passes all parity.

**`src/clustering/agglomerative.ts`:**

- Change the call site (`131`) to `const merges = nn_chain_cluster(D, n_samples, linkage)` — no `n_clusters`.
- Replace the emission-order `children_` build (`139-143`) with: (Stage A) stable distance-sort of `merges`; (Stage B) relabel Union-Find producing node ids `n..2n-2`.
- Replace the label derivation (`145-174`): apply only the first `n - n_clusters` merges _from the distance-sorted list_ through the existing Union-Find, then number components by first-seen root.
- Eliminate the double-copy at `125-128`: replace the element loop with a single `D.set(flat)` bulk copy (still float32→float64 element conversion, but JIT-friendly and one pass). Independent perf/memory win.
- Add a reducibility comment near `nn_chain_cluster`'s call documenting the single/complete/average/ward precondition.

**Callers unaffected:** `som.ts` and `find_optimal_clusters.ts` read only `labels_`/`fit_predict`; `medoid_selection.ts` depends only on `labels_`. No changes needed there.

<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->

1. Verify current agglomerative API and scipy NN-chain tie rules.
2. Replace stored nearest-neighbor merge engine with full-tree NN-chain over active index tracking.
3. Sort merges by distance, rebuild children*/distances*, and cut labels for n_clusters or distance_threshold.
4. Update linkage/unit/performance tests and fixture generator.
5. Run focused tests, type-check, lint, and update task notes/acceptance criteria.
<!-- SECTION:PLAN:END -->

## Implementation Notes

- Replaced `stored_nn_cluster(D, n, n_clusters, linkage)` with `nn_chain_cluster(D, n, linkage)`, a full-tree nearest-neighbor-chain implementation over the existing dense `Float64Array` and `Uint8Array` active-slot tracking.
- Verified SciPy 1.3.2 `_hierarchy.pyx` before coding. One task-plan hypothesis was corrected: SciPy/fastcluster order the reciprocal pair as low/high but keep the merged cluster in the higher active slot. The implementation follows that convention while returning sorted low/high merge records.
- Updated `AgglomerativeClustering` to run NN-chain to completion, then cut the sorted tree by either `n_clusters` or `distance_threshold`. Existing `distances_`, `children_`, `precomputed`, and zero-merge contracts are preserved.
- Rebuilt `children_` with a scipy-style relabel Union-Find and replaced the float32-to-float64 element copy with `D.set(flat)`.
- Updated `linkage.test.ts`, `agglomerative_perf.test.ts`, and `tools/sklearn_fixtures/generate.py`. Added a direct hub-shaped merge-loop scaling regression test; current run was 200→400 samples in 7.34ms→15.64ms (2.13x).
- Verification completed: focused agglomerative/reference/performance tests passed (50 tests); `npm run type-check` passed; `npm run lint` passed; `npx jest --runInBand --testPathIgnorePatterns=src/clustering/som_reference.test.ts` passed (52 suites, 491 tests). Full `npm test -- --runInBand` is blocked by unrelated `src/clustering/som_reference.test.ts`, which does not complete even when run alone after 10 minutes and does not exercise the agglomerative path changed here.
