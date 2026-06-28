---
id: TASK-54
title: Accelerate HDBSCAN front-half on TensorFlow.js (float32 tensor pipeline)
status: To Do
assignee: []
created_date: '2026-06-24'
updated_date: '2026-06-24'
labels:
  - performance
  - hdbscan
  - tfjs
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->

HDBSCAN is the only estimator whose compute runs entirely in plain-JS float64; it never touches the TensorFlow.js backend. This was chosen for bit-exact scikit-learn parity, but high-performance clustering is the library's first priority and parity is a verification concern that tolerates numerical drift.

The HDBSCAN pipeline splits cleanly into a parallelizable **front-half** and a sequential **tail**:

- **Front-half (move to tfjs):** dense distance matrix → per-point core (k-)distances → mutual-reachability matrix. These are `O(n²·d)` and `O(n²)` elementwise/reduction kernels that map directly onto tensor ops and dominate wall-clock at large `n` and high dimensionality.
- **Tail (stays JS):** Prim's minimum spanning tree → condensed tree → excess-of-mass → label extraction. These are inherently sequential, data-dependent graph/tree algorithms with loop-carried dependencies and union-find; they cannot be expressed on tensors without per-iteration GPU↔CPU sync that is slower than JS. This is confirmed and not negotiable.

This task moves the front-half onto the tfjs backend as a single on-tensor pipeline with **one** readback at the MST boundary, reusing the library's existing tensor distance helpers, and re-bases the parity suite for float32. The sequential tail keeps its float64 JS implementation, with a complementary path-compression cleanup of the union-find — the one JS performance win that survives the migration (the front-half JS cleanups are obsoleted because that code is replaced by tensor ops).

**Parity stance (decided):** float32 changes per-point membership _probabilities_ but not cluster _labels_ (verified empirically on the real pipeline — label equivalence held on every fixture; only the tie-free `1e-6` probability assertion overshot, by ~3e-5–1e-4). Therefore: keep label assertions strict (exact equivalence up to permutation, consistent `-1` noise), and loosen the probability tolerances to a float32-appropriate bound set from measured drift plus slack.

**Scalability note (honest framing):** the dense `O(n²)` distance + mutual-reachability matrices remain the memory ceiling regardless of backend (the benchmark hard-caps `n` at 5000 to avoid OOM). This task improves constant factors and large-`n`/high-`d` throughput; it does not raise the `O(n²)` memory wall, and the MST/condensed-tree tail remains the dominant cost at low dimensionality.

<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->

- [ ] #1 A float64-JS HDBSCAN benchmark baseline is recorded and committed before any code change, with benchmark configs extended to vary dimensionality `d` and push `n` toward the memory ceiling so the tfjs crossover is measurable; a per-fixture snapshot of `labels_`/`probabilities_` is captured as the strict-label oracle
- [ ] #2 The parity suite is re-based for float32: tie-free fixtures keep exact label equivalence (`labels_equivalent_with_noise`) but probability tolerance is loosened to a documented float32 bound; tie-bound MAE/agreement bounds are re-measured under float32 and set from observed values plus slack
- [ ] #3 The HDBSCAN front-half (distance matrix → core distances → mutual-reachability) runs on the tfjs backend, reusing `distance/pairwise_distance` and the `tf.topk` pattern from `graph/affinity`, and stays on-tensor with exactly one GPU→CPU readback at the MST input boundary
- [ ] #4 The sequential tail (`minimum_spanning_tree`, `condensation_tree`) remains float64 JS; `minimum_spanning_tree` consumes the flat readback buffer via its existing flat-array path; no tensor op is introduced into the tail
- [ ] #5 The condensation union-find `find()` is path-compressed (behavior-preserving JS performance win for the tail)
- [ ] #6 The new tensor pipeline is leak-free (tidy/dispose discipline per task-35); `dispose()`, empty input, `n === 1`, and precomputed paths all behave as before
- [ ] #7 The full HDBSCAN parity suite passes under the re-based tolerances with labels exactly matching the oracle; ESLint passes before commit
- [ ] #8 A post-migration benchmark is diffed against the baseline, quantifying the front-half speedup across `n × d`, identifying the JS↔tfjs crossover, and recording whether a small-`n` JS fallback is warranted (YAGNI-gated decision); any dead JS front-half code (full-row sort, JS distance loop, standalone `kdistance`) is deleted if no path retains it (NO BACKWARDS COMPATIBILITY)

<!-- AC:END -->

## Implementation Notes

### Investigation summary (multi-agent workflow, 2026-06-24)

A deep workflow (21 agents: map → design → adversarial-verify → synthesize) established the ground truth this plan is built on:

- **tfjs is float32-only on every backend** (cpu/wasm/webgl/node/rn). float64 parity on tfjs is structurally impossible.
- **An agent ran the real float32 pipeline.** Labels survived on every fixture; only the tie-free `1e-6` probability assertion broke (≈9.9e-5 on `nested_mcs8_ms2_*_eps0.8`, ≈3.2e-5 on `blobs_overlap_mcs10_ms2_*`). This is why the parity stance is "strict labels, relaxed probabilities".
- **MST must stay JS** (high confidence): Prim's frontier has a genuine loop-carried dependency; the `u === -1` disconnect guard and edge-recording force a per-iteration scalar readback, so a tensor MST is _slower_ at every in-scope `n`, on top of float32 breaking edge tie-ordering.
- **Condensed-tree / EoM / extract_labels must stay JS** (high confidence): top-down BFS with data-dependent branching, bottom-up stability carry, per-point variable-length ancestor climbs — no worthwhile vectorization.
- **The `O(n²)` dense-matrix memory wall is unchanged** by any backend move; the benchmark caps `n` at 5000 for this reason.

The workflow originally recommended _no_ tfjs migration and pure-JS cleanups, because it treated the strict `1e-6` tolerance as immovable and anchored on tiny fixture sizes. With the probability tier explicitly relaxed and large-`n`/high-`d` throughput as the goal, the front-half migration becomes the correct, useful application of tfjs — which this task pursues, while keeping the workflow's verified "tail stays JS" boundary and its surviving union-find cleanup.

### Design

```
INPUT X (or precomputed D)
  │  upload to Tensor2D
  ▼
[tfjs]  distance matrix        pairwise_distance_matrix(points, metric)   (54.3)
  ▼  (stays on tensor)
[tfjs]  core distances         tf.topk(-D, min_samples) → column k-1      (54.4)
  ▼  (stays on tensor)
[tfjs]  mutual-reachability    maximum(maximum(core_col, core_row), D)    (54.5)
  ▼  ── single .data() readback → flat Float32Array ──
[ JS ]  minimum_spanning_tree  (existing flat-array Prim path)            (54.6)
  ▼
[ JS ]  condensed tree / EoM / extract_labels  (+ path-compressed find)  (54.7)
  ▼
labels_, probabilities_, exemplar_indices_
```

Reuse: `pairwise_distance_matrix` / `pairwise_euclidean_matrix` (`src/distance/pairwise_distance.ts`); the `tf.topk(neg_dists, k)` smallest-k idiom (`src/graph/affinity.ts:165`); `minimum_spanning_tree`'s existing `Float64Array | number[][]` flat path (`src/graph/minimum_spanning_tree.ts`).

### Risks to watch

- **Gram-matrix cancellation:** `pairwise_euclidean_matrix` uses `‖x‖²+‖y‖²−2xᵀy`, which loses precision at the near-zero distances core/k-distance depend on (the existing `maximum(·,0)` clamp masks negatives). Labels survived in the probe, so reuse is the default; if a fixture's labels flip, fall back to a direct (non-gram) euclidean for HDBSCAN.
- **Precomputed-cosine fixture:** tie-bound, MAE already 0.150 vs the 0.16 bound. float32 may push it over — expected under the relaxed stance; re-measure and set the bound from observed + slack in 54.2.
- **Subtasks 54.3 and 54.4 are incremental:** on their own they read D back immediately (no perf benefit yet) but keep tests green; the front-half speedup is realized in 54.5 when the readback moves to the MST boundary.
