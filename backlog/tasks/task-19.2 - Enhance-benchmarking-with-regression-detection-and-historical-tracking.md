---
id: task-19.2
title: Enhance benchmarking with regression detection and historical tracking
status: To Do
assignee: []
created_date: '2025-07-29'
updated_date: '2025-07-30'
labels: []
dependencies: []
parent_task_id: task-19
---

## Description

Add performance regression detection and historical tracking to the benchmarking system to monitor performance changes over time and alert on degradations

## Acceptance Criteria

- [ ] Performance regression detection implemented with configurable thresholds
- [ ] Historical performance data stored and tracked across commits
- [ ] Automated alerts for performance degradations in CI
- [ ] Performance trend visualization over time
- [ ] ARI accuracy scores included in benchmark metrics
- [ ] Documentation updated with performance tracking guide
- [ ] CI/CD docs updated with regression detection setup
- [ ] Benchmark interpretation guide added to docs

> **Deferred:** Sklearn baseline timing comparison was originally listed here but
> requires a separate Python timing harness (out of scope for the JS-side
> regression infrastructure). It has been split into its own follow-up task.

## Implementation Plan (the how)

### Approach

Adopt [`benchmark-action/github-action-benchmark`](https://github.com/benchmark-action/github-action-benchmark)
rather than hand-rolling storage, threshold comparison, and charting. The action
handles four acceptance criteria out of the box, so the only custom code is a
results emitter, ARI integration, workflow wiring, and docs.

| Acceptance criterion | Covered by |
| --- | --- |
| Historical data across commits | History auto-committed to a `gh-pages` branch (`data.js`) |
| Trend visualization over time | Trend chart page auto-published on gh-pages |
| Regression detection w/ thresholds | `alert-threshold` config |
| Automated CI alerts | `comment-on-alert`, `fail-on-alert`, `alert-comment-cc-users` (@mention) |

### Current state (starting point)

- Working multi-backend harness: `benchmark_algorithm`, `run_benchmark_suite`,
  `format_benchmark_results` in `benchmarks/index.ts`.
- CI (`.github/workflows/benchmark.yml`) runs on PRs and posts a results comment,
  but keeps **no** history, has **no** regression detection / alerts / trend
  charts.
- `BenchmarkResult.accuracy` is declared but never populated; `make_blobs`
  returns ground-truth `y` that the harness currently discards.

### Phase 1 — Emit results in the action's format + integrate ARI

github-action-benchmark requires one metric direction per file, so emit two:

1. **ARI integration** (`benchmarks/index.ts`): destructure `{ X, y }` from
   `make_blobs`; after each fit, compute `adjusted_rand_index(y, _labels)` and
   populate the existing `accuracy` field. (AC: ARI scores in metrics.)
2. **New emitter** `benchmarks/report.ts` (+ colocated `report.test.ts`):
   - `to_perf_series(results)` → `customSmallerIsBetter` array, entries like
     `{ name: "kmeans/cpu/medium · time", unit: "ms", value }` plus
     `… · memory` (MB). Smaller = better.
   - `to_quality_series(results)` → `customBiggerIsBetter` array,
     `{ name: "kmeans/cpu/medium · ARI", unit: "ARI", value }`. Bigger = better.
3. Update `scripts/benchmark.ts` to also write `benchmarks/output/perf.json` and
   `benchmarks/output/quality.json` (keep existing YAML/MD for the human PR
   comment).
4. **Noise control:** define a single deterministic *tracked* config (one Node
   version, `cpu` backend, fixed datasets) that feeds the historical series; the
   full backend × Node matrix stays informational only. GitHub-hosted runners
   are noisy (±20–50% on timing) so only one stable config is authoritative.

### Phase 2 — Wire CI (`.github/workflows/benchmark.yml`)

- **On `push` to `main`** (the tracked series): run the deterministic config,
  then two `github-action-benchmark` steps (perf = `customSmallerIsBetter`,
  quality = `customBiggerIsBetter`), each `auto-push: true` to `gh-pages`,
  `alert-threshold: '150%'`, `comment-on-alert: true`,
  `fail-on-alert: false` initially (flip to `true` once baselines stabilize),
  `alert-comment-cc-users: '@CRJFisher'`.
- **On `pull_request`**: run the same config with `save-data-file: false` +
  `comment-always: true` → comparison-only comment, no history pollution, safe on
  forks. Keep the existing rich MD comment.
- One-time: enable a `gh-pages` branch (documented in the guide).

### Phase 3 — Documentation

New `docs/benchmarking.md` (canonical, self-contained), three sections:

- **Performance tracking guide** — how the series works, where the trend chart
  lives, how to run locally.
- **CI / regression-detection setup** — gh-pages requirement, threshold meaning
  and tuning, how alerts surface.
- **Benchmark interpretation guide** — reading time/memory/ARI series, expected
  per-algorithm ranges, runner-noise caveat & why only the tracked config is
  authoritative.

Update `README.md` Performance section to link `docs/benchmarking.md` and the
live trend page.

### Phase 4 — Verify & finalize

- `npm run benchmark` locally → confirm both JSON files validate against the
  action schema and ARI values are sane.
- `npm run lint` + run benchmark/report tests.
- Check all 8 ACs, add Implementation Notes, set status Done.

### Files

- **Edit:** `benchmarks/index.ts`, `scripts/benchmark.ts`,
  `.github/workflows/benchmark.yml`, `README.md`
- **New:** `benchmarks/report.ts`, `benchmarks/report.test.ts`,
  `docs/benchmarking.md`

### Risk

GitHub-hosted runners are noisy. Mitigated by a single fixed tracked config plus
a loose 150% threshold and `fail-on-alert: false`, tightening once baseline
history accumulates. Hard CI gating from day one would need a dedicated/self-hosted
runner — out of scope.
