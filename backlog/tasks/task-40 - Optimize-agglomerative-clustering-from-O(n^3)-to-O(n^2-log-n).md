---
id: task-40
title: Optimize agglomerative clustering from O(n^3) to O(n^2 log n)
status: To Do
assignee: []
created_date: '2026-03-20'
labels: []
---

## Description

The agglomerative clustering uses a brute-force O(n^3) nearest-pair search with Array.splice() for distance matrix contraction adding O(n^2) per merge step. This makes it impractical for datasets beyond ~500 samples. The implementation should use a min-heap/priority queue for nearest-pair tracking, and for single/complete/average/ward linkage specifically, the NNCHAIN algorithm can achieve O(n^2) amortized complexity. The distance matrix contraction should use index-based active tracking instead of physically removing rows/columns.

## Acceptance Criteria

- [ ] Agglomerative clustering uses priority queue or NNCHAIN for merge step
- [ ] Distance matrix uses index-based active tracking instead of Array.splice
- [ ] 1000-sample dataset completes in under 5 seconds
- [ ] Existing reference tests still pass
- [ ] Benchmark added for 1000 and 5000 sample datasets

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

**Caveat — in-progress feature in the working tree.** An unfinished `distance_threshold` / `distances_` / `'precomputed'`-metric feature is mid-flight in `agglomerative.ts` (currently producing TypeScript errors: a `'precomputed'` member added to the metric union that downstream signatures don't accept, and an optional `n_clusters` that is now possibly-`undefined`). This plan was written against the simpler API. Its cut logic, `children_`/`distances_` output, and the precomputed path all depend on that feature's final API shape — finalize it before implementing (see *Effort, Difficulty & Sequencing*).

## Strategy & Algorithm Choice

**Chosen backbone: NN-chain (reciprocal-nearest-neighbor) over the existing dense flat `Float64Array`, run to completion, then cut.** This is the only design that delivers a *guaranteed* O(n^2) worst-case bound, eliminating the `linkage.ts:190-192` cubic rescan storm entirely (NN-chain maintains no global per-cluster NN cache, so a merge invalidates at most the chain top, rescanned once). A heap-only variant is rejected: it accelerates global-min *selection* (O(n)→O(log n)) but leaves the destroyed-NN rebuild work untouched, so the worst case stays O(n^3) — it cannot meet the spirit of the task ("from O(n^3)").

**Reducibility — why NN-chain is exact here.** A linkage is reducible iff merging two clusters never makes the merged cluster strictly closer to any third cluster than the closer of the two originals already was. Reducibility guarantees (i) a reciprocal-NN pair is always a valid next merge, and (ii) the chain prefix stays valid after a merge (the O(n^2) bound). All four supported linkages — **single, complete, average, ward** — are Lance–Williams reducible, so NN-chain reproduces the exact agglomeration. centroid/median are *not* reducible and are not supported (`VALID_LINKAGES`, `agglomerative.ts:47-52`); a code comment guards any future addition from routing through NN-chain.

**Full-tree-then-cut contract change (mandatory).** NN-chain emits merges in *discovery* order, which is **not** distance-sorted. Consequently the current early-stop (`linkage.ts:138-140`, `target_merges = n - n_clusters`) is invalid — stopping early would merge an arbitrary subset. The new contract is:

1. Run NN-chain to completion: always exactly `n - 1` merges (the `n_clusters` parameter is removed from the merge engine).
2. Stable-sort the `n - 1` merges by `distance` ascending. V8's `Array.prototype.sort` is stable; use a numeric comparator. This sort is **non-negotiable even though `children_` is unused**, because the label cut depends on it.
3. Cut at `n_clusters` in `agglomerative.ts`.

**How label parity with sklearn is preserved.** The reference test asserts permutation-invariant partition equivalence, not exact integer labels. To produce the k-partition, apply only the **first `n - k` merges from the distance-sorted list** through the existing order-independent Union-Find (`agglomerative.ts:145-174`), then number connected components by first-seen root. Applying the `n - k` *smallest-distance* merges is exactly the dendrogram cut at `n_clusters` and is equivalent to sklearn's `_hc_cut` *for the partition*. This reuses the existing Union-Find machinery and is sufficient for parity; a full `_hc_cut` max-heap reimplementation (for exact integer labels) is **deliberately not done** — no test asserts exact integers, and it would be YAGNI surplus.

**How `children_` is handled.** `children_` has zero consumers. Rather than raise a dead field to a bit-for-bit scipy contract (rejected as surplus per the constitution), build it correctly and cheaply from the same sorted list: walk the distance-sorted merges through a small relabel Union-Find (`parent[0..n-1]`, `next_node = n`, path-halving find), pushing `[min(root_a, root_b), max(...)]` and minting node id `n + k` per merge. This is the standard scipy `label()` relabel, produces sklearn-shaped `children_` (node ids `n..2n-2` in distance order), and costs only the marginal Union-Find pass on top of the sort the labels already require. No new fixtures or exact-equality assertions are added for it.

## Refactoring Required

Per **NO BACKWARDS COMPATIBILITY**: rename and re-signature in place, update all callers, add no shims, no alias re-exports, no `n_clusters`-accepting wrapper.

**`src/clustering/linkage.ts`:**

- Replace `stored_nn_cluster(D, n, n_clusters, linkage)` with `nn_chain_cluster(D, n, linkage): MergeRecord[]`. **Drop the `n_clusters` parameter** (NN-chain always runs `n - 1` merges).
- Delete the stored-NN-only machinery: the `nn`/`nn_dist` caches (`129-130`), the init loop (`132-135`), the global-min scan (`142-150`), and the incremental NN-cache maintenance branch (`189-197`). Keep `lance_williams` (`46-71`) **verbatim** and `MergeRecord` (`31-40`) unchanged.
- Add an `Int32Array chain` stack (capacity n) + `chain_len`. Implement the reciprocal-NN loop with scipy tie rules (below).
- Preserve invariants: `cluster_a = min(a, b)`, `cluster_b = max(a, b)`; capture `ni`/`nj` *before* the Lance–Williams loop and update `size[survivor]` *after*; deactivate `removed` before the loop. The merged cluster lives under the survivor *slot* id, which must remain the find-able representative used by the relabel Union-Find.
- Rewrite the file header (`1-24`) to describe NN-chain and its guaranteed O(n^2) bound in present tense (no "previously/worst-case-O(n^3)" framing, per documentation-style).

**Distance layout decision: keep the dense flat n×n `Float64Array`.** Condensed upper-triangular storage halves the *resident* matrix but only moves the OOM peak out ~33% (the transient float32 `.data()` buffer coexists with the condensed array during the fill), and it introduces a strided-gather cache regression in the hot NN scan that threatens the 5s@1000 ceiling. The real prize is killing the O(n^3); condensed storage is a separable, benchmark-gated follow-up, deferred under YAGNI. NN-chain on dense storage proves guaranteed O(n^2) and passes all parity.

**`src/clustering/agglomerative.ts`:**

- Change the call site (`131`) to `const merges = nn_chain_cluster(D, n_samples, linkage)` — no `n_clusters`.
- Replace the emission-order `children_` build (`139-143`) with: (Stage A) stable distance-sort of `merges`; (Stage B) relabel Union-Find producing node ids `n..2n-2`.
- Replace the label derivation (`145-174`): apply only the first `n - n_clusters` merges *from the distance-sorted list* through the existing Union-Find, then number components by first-seen root.
- Eliminate the double-copy at `125-128`: replace the element loop with a single `D.set(flat)` bulk copy (still float32→float64 element conversion, but JIT-friendly and one pass). Independent perf/memory win.
- Add a reducibility comment near `nn_chain_cluster`'s call documenting the single/complete/average/ward precondition.

**Callers unaffected:** `som.ts` and `find_optimal_clusters.ts` read only `labels_`/`fit_predict`; `medoid_selection.ts` depends only on `labels_`. No changes needed there.

## Implementation Plan

1. **Download scipy reference** into `sklearn_reference/` (does not currently exist; CLAUDE.md sanctions adding it): `scipy/cluster/_hierarchy.pyx` (`nn_chain`, `label`, `LinkageUnionFind`, `condensed_index`) and `scipy/cluster/hierarchy.py` (`linkage`). Verify the tie rules **line-by-line before coding** — treat every tie rule below as a hypothesis until confirmed against the `.pyx`.

2. **Rewrite the merge engine** as `nn_chain_cluster(D, n, linkage)` in `linkage.ts`. Keep `lance_williams` and `MergeRecord`. State: `size: Float64Array(n)` (init 1), `active: Uint8Array(n)` (init 1), `chain: Int32Array(n)`, `chain_len`.

3. **Implement the chain loop** per scipy:
   - When `chain_len === 0`, push the **lowest-id active** cluster.
   - Inner loop: `a = chain[chain_len - 1]`; seed incumbent `b = chain[chain_len - 2]`, `min_dist = D[a*n+b]` if `chain_len >= 2`, else `b = -1`, `min_dist = Infinity`. Scan all active `k != a` with **strict `<`** (`if (D[a*n+k] < min_dist)`). If `chain_len >= 2 && b === chain[chain_len - 2]`, a reciprocal pair is found — break; else push `b`.

4. **Implement the merge**: pop both (`chain_len -= 2`); `survivor = min(a, b)`, `removed = max(a, b)`; capture `ni = size[survivor]`, `nj = size[removed]`; set `active[removed] = 0`; Lance–Williams update over active `k != survivor, removed` writing both `D[survivor*n+k]` and the mirror `D[k*n+survivor]`; `size[survivor] += size[removed]` after the loop; push `{cluster_a: survivor, cluster_b: removed, distance: min_dist, new_size: size[survivor]}`. Loop until `n - 1` merges recorded.

5. **Wire `agglomerative.ts`**: update the call site to drop `n_clusters`; replace the element-copy loop (`125-128`) with `D.set(flat)`.

6. **Build `children_` (Stage A + B)**: copy `merges`, stable-sort by `distance` ascending. Walk the sorted list through a relabel Union-Find (`parent: Int32Array(n)` init identity, path-halving `find`, `next_node = n`): for each row, `ra = find(cluster_a)`, `rb = find(cluster_b)`, push `[Math.min(ra, rb), Math.max(ra, rb)]`, then point both roots at the new node id `next_node++`. Assign `this.children_`.

7. **Derive `labels_` (the cut)**: take the **first `n - n_clusters`** rows of the same distance-sorted list; replay them through the existing path-compression Union-Find (`149-162`); number components 0..k-1 by first-seen root (`164-174`). Keep the `n_samples === 1` trivial branch (`105-113`) and the `n_clusters === n_samples` zero-merge case unchanged.

8. **Add the reducibility guard comment** at `nn_chain_cluster` documenting single/complete/average/ward only.

9. **Rewrite `linkage.test.ts`** to call `nn_chain_cluster` (no `n_clusters`) and assert on the **distance-sorted** post-processed output, not raw emission order:
   - Merge count: `n - 1` for n=3 → 2 merges (sorted).
   - First/second-merge distances on `base_matrix` and the 4-point single-linkage case: assert against the sorted list (these tiny cases produce the same sorted sequence as before).
   - **Delete or invert** the `stops at the requested number of clusters` test (`72-79`) — it tests the removed early-stop contract; replace it with a test that constructs `AgglomerativeClustering` with `n_clusters=2` and asserts the resulting **labels** form 2 clusters.

10. **Fix the fixture generator** (`tools/sklearn_fixtures/generate.py`): change `"nClusters"` → `"n_clusters"` (line 72) and the stale out-dir docstring (line 15) to `../../__fixtures__/agglomerative`. Do **not** add cosine or `children_`/distance fields (no consumer; out of scope).

11. **Add the adversarial worst-case test** (see Testing strategy below).

12. **Run** the full suite (reference parity, unit, perf), then **ESLint** (CLAUDE.local.md mandates lint before push; fix real errors, no `eslint-disable`, no `as any/unknown/never`). Update the `agglomerative.ts` class header (`11-18`) to present-tense NN-chain prose.

## Testing & Verification Strategy

**Reference parity (AC: "Existing reference tests still pass").** `agglomerative_reference.test.ts` globs all 13 `__fixtures__/agglomerative/*.json` and asserts `are_labelings_equivalent`. Because the label cut applies the `n - k` smallest-distance merges, the partition is identical to sklearn's; these pass with **no fixture regeneration** on the existing well-separated n=30 data. `agglomerative.test.ts` structure/validation tests (including the ward+cosine throw at `26-29` and the `n_clusters === n_samples` zero-merge case) are unaffected.

**Adversarial worst-case (the test that actually proves the win).** Fixed thresholds on well-separated `make_blobs` never exercise the stored-NN rescan path, so they cannot distinguish guaranteed O(n^2) from worst-case O(n^3). Add a **PERF-ONLY** test (no sklearn label fixture) in `agglomerative_perf.test.ts`:

- Construct a **hub/star single-linkage dataset** (one central point near-equidistant to many) — the exact `linkage.ts:190` pathology — at sizes `n` and `2n`.
- Isolate the merge loop from matrix construction where feasible; assert `t(2n)/t(n)` stays comfortably **under 8** (a generous "no cubic blowup" ceiling, e.g. `< 6`), not a tight `~4`, to tolerate JIT warmup and GC noise.
- **Do not generate sklearn labels for this dataset**: ties/near-ties are pervasive by construction, so a parity fixture on it would be flaky. It validates complexity and worst-case *correctness* (same partition as the old engine on the pathological input), not sklearn parity.

**Perf benchmarks (AC: "1000-sample under 5s", "Benchmark for 1000 and 5000").** `agglomerative_perf.test.ts` already has the 1000<5s (`6-31`) and 5000<60s (`33-61`) ward-on-blobs tests — these satisfy the AC and must keep passing. The `D.set(flat)` change and removal of the cubic path only help. Do **not** add a 10000-sample benchmark: the dense ~800 MB matrix OOMs in CI, consistent with the HDBSCAN ~5k cap (commit af45f44).

**Fixture generation** (only if regeneration is ever needed): fix the generator per step 10, then run via the committed venv: `cd tools/sklearn_fixtures && source .venv/bin/activate && python generate.py --out-dir ../../__fixtures__/agglomerative`.

**AC → verification map:**

| Acceptance Criterion | Verified by |
| --- | --- |
| Uses priority queue or NNCHAIN for merge step | NN-chain implemented in `nn_chain_cluster`; reviewed in code + `linkage.test.ts` |
| Index-based active tracking (no `Array.splice`) | Already met; preserved (`active: Uint8Array`), confirmed by code review |
| 1000 samples under 5s | `agglomerative_perf.test.ts:6-31` |
| Existing reference tests still pass | `agglomerative_reference.test.ts` (13 fixtures, no regeneration) |
| Benchmark for 1000 and 5000 | `agglomerative_perf.test.ts:6-61` (both exist) |
| (Implied) guaranteed sub-cubic | New adversarial hub-scaling ratio test |

## Effort, Difficulty & Sequencing

**This is the hardest of the open tasks — but bounded.** It is an algorithm rewrite, not a tweak. The difficulty is **not** the algorithm: NN-chain itself is ~60 lines and well-specified. The difficulty is **sklearn-parity correctness** — the full-tree-then-cut contract change, tie-breaking that must be matched line-by-line against scipy's `_hierarchy.pyx` (it cannot be reasoned out from first principles), and float-ULP fragility from the float32→float64 distance seed. The risk profile is "easy to get 90% working, hard to get the last 10% *provably* sklearn-identical," with a silent-failure trap: a wrong post-processing step can still pass the tiny fixtures by luck.

**Scope estimate:** roughly **2–4 focused sessions**. About half a session for the NN-chain core; the bulk goes to parity verification, the sort/relabel/cut post-processing, the adversarial worst-case test, and reconciling with the in-progress `distance_threshold`/`distances_`/`precomputed` feature (which roughly doubles the surface versus the original task scope).

**De-risking — do these before/while implementing:**

1. **Download scipy's `_hierarchy.pyx` reference first** (implementation step 1) and verify every tie rule against it. Do not code tie-breaking from memory.
2. **Land or finalize the in-progress `distance_threshold` / `distances_` / `'precomputed'` API before starting**, so you are not chasing a moving target. The plan's cut logic, `children_`/`distances_` output, and precomputed path all depend on that API's final shape (see the caveat in *Current State Analysis*).
3. **Decide the split.** Option A: ship (a) NN-chain core + parity first, then (b) `distance_threshold`/`precomputed` as a follow-up — smaller, independently verifiable PRs. Option B: do both together, justified because the in-progress feature work is already in the working tree. Prefer A unless the in-progress work is far enough along that separating it creates more churn than it saves.

## Risks & Mitigations

- **Tie-break parity is fragile and multi-sourced.** Strict `<` in the NN scan, chain-predecessor incumbent seed (`chain[-2]`), lowest-id chain start, and stable distance sort each independently affect merge order. *Mitigation:* verify each against the downloaded `_hierarchy.pyx` before coding (step 1); the reference test checks only partition equivalence, which is robust to tie perturbations on the well-separated fixtures.

- **Float-ULP differences (ward sqrt, average).** Seed distances come from a TF.js float32 matMul promoted to float64, while scipy is float64 throughout; near-tie merges can order differently. *Mitigation:* reuse `lance_williams` verbatim (already parity-matched by passing fixtures); assert only partition equivalence — do **not** add exact `children_`/distance assertions, which would be ULP-flaky.

- **`linkage.test.ts` emission-order assumptions break.** The `stops at requested number of clusters` test directly tests the removed early-stop. *Mitigation:* expected, not a regression — rewrite it (step 9) to assert post-cut labels via `AgglomerativeClustering`.

- **Partial migration ships wrong `children_`/labels silently.** Forgetting the distance sort before the cut yields wrong partitions that may pass tiny fixtures by luck. *Mitigation:* the sort is a single shared step feeding both `children_` and the cut; the adversarial test plus re-running all 13 fixtures are the guardrails.

- **Slot/relabel mismatch.** The relabel Union-Find's `find()` must walk from the raw `MergeRecord` *slot* ids; the merged cluster must keep living under the survivor slot id. *Mitigation:* survivor = `min(a,b)`; relabel UF seeded `parent[i]=i` for all n slots; covered by the existing 4-point and reference fixtures whose `children_` shape can be spot-checked.

- **Adversarial scaling test flakiness.** JIT/GC noise can push the ratio outside a tight band. *Mitigation:* assert a generous ceiling (`< 6`/`< 8`, "no cubic blowup"), not a tight `~4`; sanity-check the dataset triggers super-quadratic scaling against the *old* engine before landing.

- **Memory ceiling unchanged.** NN-chain removes the time pathology but the dense n×n float64 matrix still caps practical n near ~5k. *Mitigation:* out of scope for task-40's ACs (1000 & 5000); the `D.set(flat)` change trims peak/copy time. Condensed storage is a benchmark-gated follow-up.

## Open Questions

1. **Is `children_` slated to become a public/consumed output (dendrograms, threshold cuts)?** If never consumed, YAGNI argues for setting `children_ = null` and skipping the relabel Union-Find entirely (the labels cut needs only the sort, not the node-id relabel). The plan builds a correct `children_` because the marginal cost over the required sort is small, but if a human confirms it will stay dead, drop Stage B. (Note: the in-progress `distance_threshold` feature likely makes `children_`/`distances_` genuinely consumed — resolve alongside that work.)

2. **Will the TDT upgrade (branch `feature/task-49-tdt-upgrade`) add estimators that consume `AgglomerativeClustering`'s `children_` or merge distances?** None exist today. If yes, the exact-children_/distance parity work (and corresponding fixture expansion) becomes justified and should be a separate follow-up task — not folded into task-40.

3. **Should condensed upper-triangular storage be pursued as a follow-up?** It ~halves resident memory but regresses NN-scan cache locality and only modestly moves the OOM peak. Recommend deferring to a benchmark-gated follow-up rather than coupling it to this change.
