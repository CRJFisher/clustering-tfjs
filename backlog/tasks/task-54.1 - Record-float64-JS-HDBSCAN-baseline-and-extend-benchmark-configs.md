---
id: TASK-54.1
title: Record float64-JS HDBSCAN baseline and extend benchmark configs
status: Done
assignee: []
created_date: '2026-06-24'
updated_date: '2026-06-24'
labels:
  - performance
  - benchmark
dependencies: []
parent_task_id: TASK-54
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->

Establish the pre-migration float64-JS baseline that every later performance claim and parity check is measured against. No HDBSCAN benchmark baseline exists today — the only committed `benchmarks/results-*.json` predates the estimator. The current `BENCHMARK_CONFIGS` are also small-`n` and low-`d`, where the tfjs front-half cannot win (readback dominates); the suite must be extended along dimensionality `d` and toward the `O(n²)` memory ceiling so the JS↔tfjs crossover is measurable.

This subtask also captures a per-fixture snapshot of `labels_` and `probabilities_` from the unmodified pipeline, which serves as the **strict-label oracle**: later subtasks assert labels match it exactly, and that probability drift stays within the re-based bounds.

<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->

- [x] #1 The `benchmarks` HDBSCAN path runs across the existing `n <= 5000` configs and the resulting results JSON/MD is committed as the float64-JS baseline
- [x] #2 `BENCHMARK_CONFIGS` is extended with high-dimensionality cases (e.g. `d` ∈ {2, 16, 64, 128}) and larger `n` up to the documented memory ceiling, so the front-half cost is exercised where tfjs is expected to win; configs that would OOM stay skipped by the existing guard
- [x] #3 A per-fixture snapshot of `labels_` and `probabilities_` from `hdbscan.test.ts` fixtures is recorded (script + output location documented in this task's notes) to serve as the strict-label oracle for 54.7/54.9
- [x] #4 `hdbscan.test.ts`, `kdistance.test.ts`, and `condensation_tree.test.ts` are confirmed green on the unmodified baseline

<!-- AC:END -->

## Implementation Notes

### High-level summary

The pre-migration ground truth for task-54 is captured in two committed artifacts and one benchmark-config extension, with no change to the HDBSCAN pipeline itself. A focused benchmark runner records the float64-JS timing across a dimensionality and sample-size sweep, and a separate oracle script snapshots the exact `labels_`/`probabilities_` the unmodified pipeline produces on every parity fixture. Together they fix the "before" state that 54.7 (strict-label equivalence) and 54.8/54.10 (speedup diff) are measured against. A key property established here is that **HDBSCAN is backend-independent today** — it runs entirely in plain-JS float64 and never touches the tfjs backend — so the baseline is captured once on the `cpu` backend; the `tensorflow` backend only becomes meaningful once 54.3+ move the front-half onto tensors.

**Acceptance criteria addressed:**

- **AC#1** — `scripts/benchmark-hdbscan.ts` (`npm run benchmark:hdbscan`) runs the HDBSCAN path across all `n <= 5000` configs, timing each as the median of 3 runs (with min/max), and writes `benchmarks/hdbscan-baseline.{yaml,md}` (both committed). The `n = 10000` `large` config stays skipped by the same `samples > 5000` guard `run_benchmark_suite` uses.
- **AC#2** — `BENCHMARK_CONFIGS` (`benchmarks/index.ts`) gains a six-config front-half sweep: `d ∈ {2, 16, 64, 128}` at `n = 2000` to isolate the `O(n²·d)` dimensionality cost, plus `n = 5000` at `d ∈ {16, 128}` to push toward the dense-matrix memory ceiling. The baseline shows the expected gradient — `n=2000` ranges ~0.93s (d=2) → ~1.18s (d=128); `n=5000` reaches ~8.9s at d=128 — giving the later tfjs crossover a measurable range.
- **AC#3** — `scripts/hdbscan-oracle.ts` (`npm run hdbscan:oracle`) replicates the test's exact fit logic (`fixture_params`/`fit_input`, including `min_samples` null-handling and the precomputed→`distance_matrix` path) over all 33 `__fixtures__/hdbscan/*.json` fixtures and writes the strict-label oracle to **`__fixtures__/hdbscan/__oracle__/hdbscan_oracle.json`** (each entry: `file`, `tie_free`, `metric`, `labels`, `probabilities`). The `__oracle__` directory is deliberately not a `.json` sibling, so the test's non-recursive `load_fixtures` readdir filter skips it and the fixture-count assertions (e.g. `degenerate.length === 4`) stay intact.
- **AC#4** — `hdbscan.test.ts`, `kdistance.test.ts`, and `condensation_tree.test.ts` were run on the unmodified pipeline and all pass (143 tests green), confirming the baseline is sound before any migration.

**Oracle regeneration policy:** the oracle is the "before" snapshot and must be regenerated only when the fixtures themselves change — never to absorb probability drift introduced by the float32 migration. 54.7/54.9 assert labels match it exactly (up to cluster-id permutation, consistent `-1` noise) and that probability drift stays within the re-based float32 bounds.

**Review findings applied:** removed a per-row `repeats` field from the baseline rows that duplicated the run-level `repeats` metadata (surplus). The new scripts are not covered by `npm run lint` (which scopes `src test_support benchmarks`); they were linted explicitly and pass, and `scripts/` was intentionally left out of the lint scope because pre-existing debug/release scripts there carry unrelated violations.
