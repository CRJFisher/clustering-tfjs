---
id: TASK-54.10
title: Benchmark the front-half win and decide on a small-n fallback
status: To Do
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

If a size gate is adopted, model it on the existing size/mode-gated paths in the codebase and test the boundary value (the highest-risk case) on both paths. If the standalone `kdistance` helper and its tests are removed, also remove or migrate the `__fixtures__/density/kdistance_*` fixtures and the `kdistance` references in `condensation_tree.test.ts` so nothing imports deleted code.
