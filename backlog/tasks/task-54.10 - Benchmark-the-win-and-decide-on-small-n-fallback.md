---
id: TASK-54.10
title: Benchmark the front-half win and decide on a small-n fallback
status: Done
assignee: []
created_date: '2026-06-24'
updated_date: '2026-06-24'
labels:
  - hdbscan
  - benchmark
  - performance
dependencies:
  - task-54.9
parent_task_id: TASK-54
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->

Quantify the migration's payoff and close out any dead code. Re-run the extended benchmark from 54.1 on the migrated pipeline, diff against the float64-JS baseline across `n × d`, and report the front-half speedup and the JS↔tfjs crossover point. Use the measured crossover to make a YAGNI-gated decision: if tfjs regresses meaningfully at small `n` (readback overhead), decide whether a small-`n` JS fallback path is worth its maintenance cost; otherwise keep the single tensor path.

Then delete any JS front-half code that no path retains (full-row sort, the JS distance loop, the standalone `kdistance` helper if unused), per NO BACKWARDS COMPATIBILITY — no dual-shape code kept "just in case".

<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->

- [ ] #1 A post-migration benchmark is produced and diffed against the 54.1 baseline using `benchmarks/compare.ts`, reporting timing and memory across the `n × d` configs and identifying the JS↔tfjs crossover
- [ ] #2 The write-up states the front-half speedup honestly: where tfjs wins (large `n`, high `d`), where the MST/condensed-tree tail still dominates (low `d`), and that the `O(n²)` memory ceiling is unchanged
- [ ] #3 A documented decision records whether a small-`n` JS fallback is added; if added, both paths are covered by an equivalence test on the boundary value (AGENTS.md "test equivalent paths equivalently"); if not, the rationale is recorded
- [ ] #4 Any JS front-half code retained by no path is deleted; no code accepts both a JS-sort and a tensor shape behind a permanent toggle
- [ ] #5 ESLint passes before commit

<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
## High-level summary

The post-migration benchmark (`benchmarks/hdbscan-postmigration.{yaml,md}`) confirms a
dramatic front-half speedup with the `tensorflow` (tfjs-node) backend at every tested size.
No small-n JS fallback is warranted. Dead JS front-half code is deleted.

### Benchmark results (tfjs-node vs float64-JS baseline, cross-session, same machine)

| Config | n × d | float64-JS baseline (ms) | float32-tfjs-node (ms) | Speedup |
| ------ | ----- | ------------------------ | ---------------------- | ------- |
| small | 100×10 | 4.87 | 1.66 | **2.9×** |
| medium | 1000×50 | 226.95 | 36.94 | **6.1×** |
| hdbscan_n2000_d128 | 2000×128 | 1182.55 | 132.06 | **9.0×** |
| hdbscan_n5000_d16 | 5000×16 | 6284.35 | 684.45 | **9.2×** |
| hdbscan_n5000_d128 | 5000×128 | 8887.47 | 702.62 | **12.6×** |

In-session (same session, cpu TF interpreter vs tensorflow native) speedups are 8–40×.

**Where tfjs wins:** The `tensorflow` (tfjs-node) backend outperforms the old float64-JS
pipeline at every configuration, including the smallest (n=100). The `O(n²·d)` front-half
dominates at medium-to-large `n` and the native tensor kernels deliver 9–13× end-to-end
speedups in the target range.

**Where the MST tail still dominates:** At low dimensionality (d=2) the front-half work is
minimal; the `O(n²)` Prim sweep and condensed-tree traversal make up most of the runtime.
The tensor front-half still helps — it's fast — but the per-iteration JS tail caps overall
speedup at low `d`.

**O(n²) memory wall:** The dense pairwise-distance and mutual-reachability matrices remain
`O(n²)`. This migration improves constant factors and throughput; the `n ≈ 5000` dense-matrix
ceiling is unchanged.

**Tensor leak check:** Tensor count delta = 0 on every `fit` call — no dispose regression.

### Small-n JS fallback decision

**No fallback (YAGNI).** The `tensorflow` backend is faster than the float64-JS baseline even
at n=100 (1.66 ms vs 4.87 ms, 2.9× faster). The fallback threshold is >1.5× overhead AND >5 ms
absolute regression — neither fires. Adding a fallback would reintroduce the dual front-half code
this task removes.

### Dead code deleted

- `src/distance/kdistance.ts` — standalone core-distance helper; no longer on any production
  code path (production HDBSCAN uses `tf.topk` in `core_distances()`).
- `src/distance/kdistance.test.ts` — test for the deleted helper.
- `__fixtures__/density/kdistance_{small_2d,small_3d,medium_2d}.json` — fixtures for the
  deleted test.
- `src/graph/condensation_tree.test.ts` — removed the `kdistance` import; the two call sites
  inline the trivial extraction: `const k = Math.min(ms, n); const core = nd.map((row) => row[k - 1])`.

`src/graph/mutual_reachability.ts` is retained — it is the JS reference oracle for the
`mutual_reachability.test.ts` fixture suite and the condensation_tree end-to-end tests.

<!-- SECTION:NOTES:END -->
