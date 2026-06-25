#!/usr/bin/env node
/**
 * Diagnostic script for task-54.9: measures actual float32 probability drift
 * across all HDBSCAN fixtures. Run after the full front-half tfjs migration to
 * determine tight constant values for hdbscan.test.ts.
 *
 * Outputs two summary tables (tie-free, tie-bound) and a recommended constant
 * update block. Use the block to update the four named constants in
 * src/clustering/hdbscan.test.ts.
 *
 * Run with: npx ts-node scripts/measure_float32_drift.ts
 */
import fs from 'fs';
import path from 'path';

import { HDBSCAN } from '../src/clustering/hdbscan';
import type { HDBSCANParams } from '../src/clustering/types';
import {
  alignment_agreement,
  labels_equivalent_with_noise,
} from '../test_support/label_agreement';

const FIXTURE_DIR = path.join(process.cwd(), '__fixtures__', 'hdbscan');
const ORACLE_PATH = path.join(FIXTURE_DIR, '__oracle__', 'hdbscan_oracle.json');

interface HdbscanFixture {
  name: string;
  params: {
    min_cluster_size: number;
    min_samples: number | null;
    cluster_selection_method: 'eom' | 'leaf';
    cluster_selection_epsilon: number;
    metric: 'euclidean' | 'manhattan' | 'precomputed';
  };
  labels: number[];
  probabilities: number[];
  tie_free: boolean;
  X?: number[][];
  distance_matrix?: number[][];
}

interface OracleEntry {
  file: string;
  tie_free: boolean;
  labels: number[];
  probabilities: number[];
}

function fixture_params(fixture: HdbscanFixture): Partial<HDBSCANParams> {
  const params: Partial<HDBSCANParams> = {
    min_cluster_size: fixture.params.min_cluster_size,
    cluster_selection_method: fixture.params.cluster_selection_method,
    cluster_selection_epsilon: fixture.params.cluster_selection_epsilon,
    metric: fixture.params.metric,
  };
  if (fixture.params.min_samples != null) {
    params.min_samples = fixture.params.min_samples;
  }
  return params;
}

function fit_input(fixture: HdbscanFixture): number[][] {
  return fixture.params.metric === 'precomputed'
    ? fixture.distance_matrix!
    : fixture.X!;
}

function max_abs_diff(a: number[], b: number[]): number {
  let max = 0;
  for (let i = 0; i < a.length; i++) {
    const d = Math.abs(a[i] - b[i]);
    if (d > max) max = d;
  }
  return max;
}

function mean_abs_diff(a: number[], b: number[]): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += Math.abs(a[i] - b[i]);
  return s / a.length;
}

// Round up to 2 significant figures for the recommended bound.
function ceil_2sig(x: number): number {
  if (x === 0) return 0;
  const mag = Math.pow(10, Math.floor(Math.log10(x)));
  return Math.ceil(x / mag * 10) / 10 * mag;
}

async function main(): Promise<void> {
  const oracle_entries: OracleEntry[] = JSON.parse(
    fs.readFileSync(ORACLE_PATH, 'utf-8'),
  ) as OracleEntry[];
  const oracle_map = new Map<string, OracleEntry>(
    oracle_entries.map((e) => [e.file, e]),
  );

  const files = fs
    .readdirSync(FIXTURE_DIR)
    .filter((f) => f.endsWith('.json'))
    .sort();

  interface TieFreeRow {
    file: string;
    max_drift_vs_sklearn: number;
    max_drift_vs_oracle: number;
    labels_ok_sklearn: boolean;
    labels_ok_oracle: boolean;
  }
  interface TieBoundRow {
    file: string;
    mae_vs_sklearn: number;
    mae_vs_oracle: number;
    agree_vs_sklearn: number;
    agree_vs_oracle: number;
    clusters_ok: boolean;
    labels_ok_oracle: boolean;
  }

  const tie_free_rows: TieFreeRow[] = [];
  const tie_bound_rows: TieBoundRow[] = [];

  for (const file of files) {
    const fixture = JSON.parse(
      fs.readFileSync(path.join(FIXTURE_DIR, file), 'utf-8'),
    ) as HdbscanFixture;

    const oracle = oracle_map.get(file);
    if (!oracle) {
      console.warn(`  WARNING: no oracle entry for ${file}`);
    }

    const model = new HDBSCAN(fixture_params(fixture));
    await model.fit(fit_input(fixture));
    const probs = model.probabilities_!;
    const labels = model.labels_!;
    model.dispose();

    if (fixture.tie_free) {
      tie_free_rows.push({
        file,
        max_drift_vs_sklearn: max_abs_diff(probs, fixture.probabilities),
        max_drift_vs_oracle: oracle
          ? max_abs_diff(probs, oracle.probabilities)
          : NaN,
        labels_ok_sklearn: labels_equivalent_with_noise(labels, fixture.labels),
        labels_ok_oracle: oracle
          ? labels_equivalent_with_noise(labels, oracle.labels)
          : false,
      });
    } else {
      const mine_clusters = new Set(labels.filter((l) => l !== -1)).size;
      const sk_clusters = new Set(fixture.labels.filter((l) => l !== -1)).size;
      tie_bound_rows.push({
        file,
        mae_vs_sklearn: mean_abs_diff(probs, fixture.probabilities),
        mae_vs_oracle: oracle
          ? mean_abs_diff(probs, oracle.probabilities)
          : NaN,
        agree_vs_sklearn: alignment_agreement(labels, fixture.labels),
        agree_vs_oracle: oracle
          ? alignment_agreement(labels, oracle.labels)
          : NaN,
        clusters_ok: mine_clusters === sk_clusters,
        labels_ok_oracle: oracle
          ? labels_equivalent_with_noise(labels, oracle.labels)
          : false,
      });
    }
    process.stdout.write('.');
  }
  console.log('\n');

  // --- TIE-FREE TABLE ---
  console.log('=== TIE-FREE FIXTURES ===');
  console.log(
    'file'.padEnd(55),
    'max_drift_sklearn'.padStart(20),
    'max_drift_oracle'.padStart(18),
    'labels_ok(sk)'.padStart(15),
    'labels_ok(or)'.padStart(15),
  );
  console.log('-'.repeat(125));
  for (const r of tie_free_rows.sort(
    (a, b) => b.max_drift_vs_sklearn - a.max_drift_vs_sklearn,
  )) {
    console.log(
      r.file.padEnd(55),
      r.max_drift_vs_sklearn.toExponential(3).padStart(20),
      r.max_drift_vs_oracle.toExponential(3).padStart(18),
      (r.labels_ok_sklearn ? 'YES' : 'NO !!!').padStart(15),
      (r.labels_ok_oracle ? 'YES' : 'NO !!!').padStart(15),
    );
  }

  // --- TIE-BOUND TABLE ---
  console.log('\n=== TIE-BOUND FIXTURES ===');
  console.log(
    'file'.padEnd(55),
    'MAE_sklearn'.padStart(14),
    'MAE_oracle'.padStart(12),
    'agree_sk'.padStart(10),
    'agree_or'.padStart(10),
    'clust_ok'.padStart(10),
    'lbl_ok(or)'.padStart(12),
  );
  console.log('-'.repeat(127));
  for (const r of tie_bound_rows.sort(
    (a, b) => b.mae_vs_sklearn - a.mae_vs_sklearn,
  )) {
    console.log(
      r.file.padEnd(55),
      r.mae_vs_sklearn.toFixed(4).padStart(14),
      r.mae_vs_oracle.toFixed(4).padStart(12),
      r.agree_vs_sklearn.toFixed(4).padStart(10),
      r.agree_vs_oracle.toFixed(4).padStart(10),
      (r.clusters_ok ? 'YES' : 'NO !!!').padStart(10),
      (r.labels_ok_oracle ? 'YES' : 'NO !!!').padStart(12),
    );
  }

  // --- RECOMMENDATIONS ---
  const max_tie_free_drift = Math.max(
    ...tie_free_rows.map((r) => r.max_drift_vs_sklearn),
  );
  const worst_tie_free = tie_free_rows.find(
    (r) => r.max_drift_vs_sklearn === max_tie_free_drift,
  )!;

  const max_tie_bound_mae = Math.max(
    ...tie_bound_rows.map((r) => r.mae_vs_sklearn),
  );
  const worst_tie_bound_mae = tie_bound_rows.find(
    (r) => r.mae_vs_sklearn === max_tie_bound_mae,
  )!;

  const min_tie_bound_agree = Math.min(
    ...tie_bound_rows.map((r) => r.agree_vs_sklearn),
  );
  const worst_tie_bound_agree = tie_bound_rows.find(
    (r) => r.agree_vs_sklearn === min_tie_bound_agree,
  )!;

  // Proposed bounds: observed * 1.5, rounded up to 2 significant figures.
  const proposed_tie_free_atol = ceil_2sig(max_tie_free_drift * 1.5);
  // MAE: add 20% headroom.
  const proposed_mae_max = Math.ceil(max_tie_bound_mae * 1.2 * 100) / 100;
  // Agreement: subtract headroom (floor at observed - 0.03).
  const proposed_agree_min =
    Math.floor((min_tie_bound_agree - 0.03) * 100) / 100;

  console.log('\n=== RECOMMENDED CONSTANT UPDATE (src/clustering/hdbscan.test.ts) ===');
  console.log(
    `TIE_FREE_PROB_ATOL:      current=1e-3  →  proposed=${proposed_tie_free_atol.toExponential(1)}`,
    `  (observed max: ${max_tie_free_drift.toExponential(3)}, fixture: ${worst_tie_free.file})`,
  );
  console.log(
    `TIE_BOUND_MAE_MAX:       current=0.18  →  proposed=${proposed_mae_max.toFixed(2)}`,
    `  (observed max: ${max_tie_bound_mae.toFixed(4)}, fixture: ${worst_tie_bound_mae.file})`,
  );
  console.log(
    `TIE_BOUND_AGREEMENT_MIN: current=0.94  →  proposed=${proposed_agree_min.toFixed(2)}`,
    `  (observed min: ${min_tie_bound_agree.toFixed(4)}, fixture: ${worst_tie_bound_agree.file})`,
  );

  const label_mismatches_vs_sklearn = [
    ...tie_free_rows.filter((r) => !r.labels_ok_sklearn),
    ...tie_bound_rows.filter((r) => !r.clusters_ok),
  ];
  if (label_mismatches_vs_sklearn.length > 0) {
    console.log(
      '\n!!! LABEL MISMATCHES DETECTED — treat as a regression, not a tolerance issue !!!',
    );
    for (const r of label_mismatches_vs_sklearn) {
      console.log('  ', r.file);
    }
  } else {
    console.log(
      '\n✓ Labels match sklearn oracle on all fixtures (up to cluster-id permutation).',
    );
  }
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
