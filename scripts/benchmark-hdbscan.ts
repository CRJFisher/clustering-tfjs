#!/usr/bin/env node
/**
 * Records the float64-JS HDBSCAN performance baseline for task-54.
 *
 * HDBSCAN currently runs entirely in plain-JS float64 and never touches the
 * tfjs backend, so its timing is backend-independent today; this baseline is
 * captured on the `cpu` backend as the single JS reference. Task-54.3+ move the
 * front-half (distance matrix → core distances → mutual reachability) onto the
 * tfjs backend, at which point the post-migration benchmark records the
 * `tensorflow` backend and is diffed against this file (task-54.8 / 54.10).
 *
 * Each (config) is timed `REPEATS` times and the median execution time is
 * reported so the baseline is not skewed by a single noisy run. Configs above
 * the dense-O(n²) ceiling (n > 5000) are skipped, mirroring run_benchmark_suite.
 *
 * Output: `benchmarks/hdbscan-baseline.{yaml,md}`
 * Run with: `npm run benchmark:hdbscan`
 */
import { writeFileSync } from 'fs';
import { join } from 'path';
import * as yaml from 'js-yaml';

import { benchmark_algorithm, BENCHMARK_CONFIGS, BenchmarkResult } from '../benchmarks';

const REPEATS = 3;
const HDBSCAN_MAX_SAMPLES = 5000;
const BACKEND = 'cpu';

interface HdbscanBaselineRow {
  label: string;
  dataset_size: number;
  features: number;
  median_time_ms: number;
  min_time_ms: number;
  max_time_ms: number;
  memory_used_mb: number;
}

function median(values: number[]): number {
  const sorted = [...values].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 0
    ? (sorted[mid - 1] + sorted[mid]) / 2
    : sorted[mid];
}

function format_baseline_table(rows: HdbscanBaselineRow[]): string {
  const header =
    '| Config | n × d | Median (ms) | Min (ms) | Max (ms) | Memory (MB) |\n' +
    '|--------|-------|-------------|----------|----------|-------------|';
  const body = rows
    .map(
      (r) =>
        `| ${r.label} | ${r.dataset_size}×${r.features} | ${r.median_time_ms.toFixed(2)} | ` +
        `${r.min_time_ms.toFixed(2)} | ${r.max_time_ms.toFixed(2)} | ${r.memory_used_mb.toFixed(2)} |`,
    )
    .join('\n');
  return `${header}\n${body}`;
}

async function main(): Promise<void> {
  console.log(
    `Recording float64-JS HDBSCAN baseline on '${BACKEND}' backend (${REPEATS} repeats, median reported)...\n`,
  );

  const rows: HdbscanBaselineRow[] = [];

  for (const config of BENCHMARK_CONFIGS) {
    if (config.samples > HDBSCAN_MAX_SAMPLES) {
      console.log(
        `Skipping ${config.label} (n=${config.samples} exceeds the dense O(n²) ceiling).`,
      );
      continue;
    }

    const times: number[] = [];
    let last: BenchmarkResult | null = null;
    for (let i = 0; i < REPEATS; i++) {
      const result = await benchmark_algorithm('hdbscan', config, BACKEND);
      times.push(result.execution_time);
      last = result;
    }

    const row: HdbscanBaselineRow = {
      label: config.label,
      dataset_size: config.samples,
      features: config.features,
      median_time_ms: median(times),
      min_time_ms: Math.min(...times),
      max_time_ms: Math.max(...times),
      memory_used_mb: last!.memory_used / 1024 / 1024,
    };
    rows.push(row);

    console.log(
      `  ${config.label} (${config.samples}×${config.features}): ` +
        `median ${row.median_time_ms.toFixed(2)}ms`,
    );
  }

  const yaml_path = join(process.cwd(), 'benchmarks', 'hdbscan-baseline.yaml');
  const payload = {
    algorithm: 'hdbscan',
    pipeline: 'float64-js',
    backend: BACKEND,
    repeats: REPEATS,
    note: 'HDBSCAN runs entirely in float64 JS and is backend-independent; this is the pre-migration baseline for task-54.',
    results: rows,
  };
  writeFileSync(
    yaml_path,
    yaml.dump(payload, { indent: 2, lineWidth: -1, noRefs: true, sortKeys: false }),
  );

  const md_path = yaml_path.replace('.yaml', '.md');
  const md = `# HDBSCAN float64-JS Baseline (task-54)

Pre-migration baseline of the all-JS HDBSCAN pipeline, captured on the \`${BACKEND}\`
backend (median of ${REPEATS} runs per config). HDBSCAN is backend-independent at
this point; task-54.3+ move the front-half onto tfjs and the post-migration run
is diffed against this file.

## Results

${format_baseline_table(rows)}
`;
  writeFileSync(md_path, md);

  console.log(`\nWrote ${rows.length} rows to:`);
  console.log(`  ${yaml_path}`);
  console.log(`  ${md_path}`);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
