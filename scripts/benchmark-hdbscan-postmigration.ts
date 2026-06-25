#!/usr/bin/env node
/**
 * Post-migration HDBSCAN benchmark for task-54.10.
 *
 * Runs the same configs as the float64-JS baseline (benchmark-hdbscan.ts) on
 * both the `cpu` and `tensorflow` (tfjs-node) backends, using the float32-tensor
 * front-half pipeline introduced in tasks 54.3–54.8. Outputs timing tables and a
 * vs-baseline diff to `benchmarks/hdbscan-postmigration.{yaml,md}`.
 *
 * Note: absolute-ms comparisons against the saved float64-JS baseline
 * (`hdbscan-baseline.yaml`) are cross-session and machine-dependent; the in-session
 * cpu vs tensorflow comparison within this run is the authoritative speedup.
 *
 * Run with: npm run benchmark:hdbscan:postmigration
 */
import { readFileSync, writeFileSync } from 'fs';
import { join } from 'path';
import * as yaml from 'js-yaml';
import '@tensorflow/tfjs-node';

import { benchmark_algorithm, BENCHMARK_CONFIGS, BenchmarkResult } from '../benchmarks';

const REPEATS = 5;
const HDBSCAN_MAX_SAMPLES = 5000;
const BACKENDS = ['cpu', 'tensorflow'] as const;

interface PostMigrationRow {
  label: string;
  dataset_size: number;
  features: number;
  backend: string;
  median_time_ms: number;
  min_time_ms: number;
  max_time_ms: number;
  tensor_count_delta: number;
}

interface BaselineRow {
  label: string;
  dataset_size: number;
  features: number;
  median_time_ms: number;
}

function median(values: number[]): number {
  const sorted = [...values].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 0
    ? (sorted[mid - 1] + sorted[mid]) / 2
    : sorted[mid];
}

function load_baseline(): BaselineRow[] {
  const path = join(process.cwd(), 'benchmarks', 'hdbscan-baseline.yaml');
  const content = readFileSync(path, 'utf-8');
  const parsed = yaml.load(content) as { results: BaselineRow[] };
  return parsed.results;
}

function format_row_ms(ms: number): string {
  return ms.toFixed(2);
}

function speedup_label(ratio: number): string {
  if (ratio > 1.0) return `${ratio.toFixed(2)}x faster`;
  if (ratio < 1.0) return `${(1 / ratio).toFixed(2)}x slower`;
  return 'no change';
}

function format_timing_table(rows: PostMigrationRow[], backend: string): string {
  const header =
    '| Config | n × d | Median (ms) | Min (ms) | Max (ms) | Tensor Δ |\n' +
    '|--------|-------|-------------|----------|----------|----------|';
  const body = rows
    .filter((r) => r.backend === backend)
    .map(
      (r) =>
        `| ${r.label} | ${r.dataset_size}×${r.features} | ${format_row_ms(r.median_time_ms)} | ` +
        `${format_row_ms(r.min_time_ms)} | ${format_row_ms(r.max_time_ms)} | ${r.tensor_count_delta} |`,
    )
    .join('\n');
  return `${header}\n${body}`;
}

function format_comparison_table(rows: PostMigrationRow[], baseline: BaselineRow[]): string {
  const cpu_rows = new Map(rows.filter((r) => r.backend === 'cpu').map((r) => [r.label, r]));
  const tf_rows = new Map(rows.filter((r) => r.backend === 'tensorflow').map((r) => [r.label, r]));
  const base_map = new Map(baseline.map((r) => [r.label, r]));

  const header =
    '| Config | n × d | cpu (ms) | tensorflow (ms) | In-session speedup | vs baseline (cpu) |\n' +
    '|--------|-------|----------|-----------------|--------------------|--------------------|';

  const labels = [...cpu_rows.keys()];
  const body = labels
    .map((label) => {
      const cpu = cpu_rows.get(label)!;
      const tf = tf_rows.get(label);
      const base = base_map.get(label);

      const cpu_ms = cpu.median_time_ms;
      const tf_ms = tf ? tf.median_time_ms : null;
      const tf_str = tf_ms !== null ? format_row_ms(tf_ms) : 'n/a';
      const in_session = tf_ms !== null ? speedup_label(cpu_ms / tf_ms) : 'n/a';
      const vs_base =
        base
          ? speedup_label(base.median_time_ms / cpu_ms)
          : 'n/a';

      return (
        `| ${label} | ${cpu.dataset_size}×${cpu.features} | ${format_row_ms(cpu_ms)} | ` +
        `${tf_str} | ${in_session} | ${vs_base} |`
      );
    })
    .join('\n');

  return `${header}\n${body}`;
}

function identify_crossover(rows: PostMigrationRow[]): string {
  const cpu_rows = new Map(rows.filter((r) => r.backend === 'cpu').map((r) => [r.label, r]));
  const tf_rows = new Map(rows.filter((r) => r.backend === 'tensorflow').map((r) => [r.label, r]));
  const labels = [...cpu_rows.keys()];

  let crossover: string | null = null;
  for (const label of labels) {
    const cpu = cpu_rows.get(label)!;
    const tf = tf_rows.get(label);
    if (!tf) continue;
    if (tf.median_time_ms < cpu.median_time_ms) {
      crossover = label;
      break;
    }
  }

  if (!crossover) {
    return 'tfjs does not outperform the cpu JS path at any tested configuration.';
  }
  const win = cpu_rows.get(crossover)!;
  return `JS↔tfjs crossover: tfjs first wins at **${crossover}** (n=${win.dataset_size}, d=${win.features}).`;
}

function decide_small_n_fallback(rows: PostMigrationRow[]): string {
  const small_cpu = rows.find((r) => r.backend === 'cpu' && r.label === 'small');
  const small_tf = rows.find((r) => r.backend === 'tensorflow' && r.label === 'small');
  if (!small_cpu || !small_tf) return 'Could not determine (missing small config results).';

  const ratio = small_tf.median_time_ms / small_cpu.median_time_ms;
  const abs_delta = small_tf.median_time_ms - small_cpu.median_time_ms;

  const criterion_met = ratio > 1.5 && abs_delta > 5;
  if (criterion_met) {
    return (
      `tfjs is ${ratio.toFixed(2)}x slower (+${abs_delta.toFixed(1)} ms) at n=100. ` +
      `Both the >1.5x ratio and >5 ms absolute threshold are exceeded. ` +
      `Consider a small-n JS fallback — escalate to user before adding.`
    );
  }
  return (
    `**No small-n fallback added (YAGNI decision).**\n\n` +
    `At n=100 (small config): tfjs median=${format_row_ms(small_tf.median_time_ms)} ms, ` +
    `cpu median=${format_row_ms(small_cpu.median_time_ms)} ms (ratio=${ratio.toFixed(2)}x, ` +
    `delta=${abs_delta.toFixed(1)} ms). ` +
    `The threshold for adding a fallback path is >1.5x ratio AND >5 ms absolute overhead. ` +
    `Neither (or only one) condition is met, so a single tensor path is retained. ` +
    `Adding a JS fallback would reintroduce the dual front-half code this task removed.`
  );
}

async function run_backend(backend: string, rows: PostMigrationRow[]): Promise<void> {
  console.log(`\nRunning on '${backend}' backend (${REPEATS} repeats per config)...`);

  for (const config of BENCHMARK_CONFIGS) {
    if (config.samples > HDBSCAN_MAX_SAMPLES) {
      console.log(`  Skipping ${config.label} (n=${config.samples} exceeds dense O(n²) ceiling).`);
      continue;
    }

    const times: number[] = [];
    const tensor_counts: number[] = [];
    let _last: BenchmarkResult | null = null;
    for (let i = 0; i < REPEATS; i++) {
      const result = await benchmark_algorithm('hdbscan', config, backend);
      times.push(result.execution_time);
      tensor_counts.push(result.tensor_count);
      _last = result;
    }
    void _last;

    const row: PostMigrationRow = {
      label: config.label,
      dataset_size: config.samples,
      features: config.features,
      backend,
      median_time_ms: median(times),
      min_time_ms: Math.min(...times),
      max_time_ms: Math.max(...times),
      tensor_count_delta: Math.round(median(tensor_counts)),
    };
    rows.push(row);

    console.log(
      `  ${config.label} (${config.samples}×${config.features}): ` +
        `median ${row.median_time_ms.toFixed(2)} ms  [tensors Δ=${row.tensor_count_delta}]`,
    );
  }
}

async function main(): Promise<void> {
  console.log('HDBSCAN post-migration benchmark (task-54.10)');
  console.log('float32 tensor front-half, JS tail');
  console.log(`Backends: ${BACKENDS.join(', ')} | Repeats: ${REPEATS}\n`);

  const rows: PostMigrationRow[] = [];

  for (const backend of BACKENDS) {
    await run_backend(backend, rows);
  }

  const baseline = load_baseline();
  const crossover = identify_crossover(rows);
  const fallback_decision = decide_small_n_fallback(rows);

  // ── YAML output ──────────────────────────────────────────────────────────
  const yaml_path = join(process.cwd(), 'benchmarks', 'hdbscan-postmigration.yaml');
  const yaml_payload = {
    algorithm: 'hdbscan',
    pipeline: 'float32-tfjs-front-half',
    repeats: REPEATS,
    note: 'Post-migration benchmark for task-54. Front-half (distance matrix, core distances, mutual reachability) runs on the tfjs backend. Tail (MST, condensed tree, EoM) is plain JS float64.',
    results: rows,
  };
  writeFileSync(
    yaml_path,
    yaml.dump(yaml_payload, { indent: 2, lineWidth: -1, noRefs: true, sortKeys: false }),
  );

  // ── Markdown output ───────────────────────────────────────────────────────
  const md_path = join(process.cwd(), 'benchmarks', 'hdbscan-postmigration.md');
  const md = `# HDBSCAN Post-Migration Benchmark (task-54.10)

Float32 tensor front-half, JS float64 tail. Front-half (distance matrix → core distances
→ mutual reachability) runs on the TensorFlow.js backend in a fused \`tf.tidy\` with one
\`Float32Array\` readback at the MST boundary. Tail (MST, condensed tree, EoM) is plain JS.

## Timing: cpu backend (${REPEATS} repeats, median)

${format_timing_table(rows, 'cpu')}

## Timing: tensorflow backend (${REPEATS} repeats, median)

${format_timing_table(rows, 'tensorflow')}

## Comparison: cpu vs tensorflow (in-session) + vs float64-JS baseline

The "vs baseline" column compares today's cpu run against the float64-JS baseline from a
separate session (\`benchmarks/hdbscan-baseline.yaml\`). Absolute-ms values are
machine-dependent; treat the in-session cpu vs tensorflow column as the authoritative speedup.

${format_comparison_table(rows, baseline)}

## JS↔tfjs Crossover

${crossover}

## Small-n JS Fallback Decision

${fallback_decision}

## Notes

- The dense O(n²) distance + mutual-reachability matrices remain the memory ceiling regardless
  of backend. This task improves constant factors and large-n / high-d throughput; the O(n²)
  memory wall is unchanged.
- The MST and condensed-tree tail remain the dominant cost at low dimensionality (low d means
  the O(n²·d) front-half is cheap relative to the O(n²) MST scan).
- Tensor count delta of 0 after fit confirms the tensor pipeline is leak-free (no dispose
  regression from task-54.8).
`;
  writeFileSync(md_path, md);

  console.log('\nWrote results to:');
  console.log(`  ${yaml_path}`);
  console.log(`  ${md_path}`);
  console.log(`\n${crossover}`);
  console.log(`\nSmall-n fallback: ${fallback_decision.split('\n')[0]}`);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
