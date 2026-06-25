#!/usr/bin/env node
/**
 * Records the strict-label oracle for task-54: the per-fixture `labels_` and
 * `probabilities_` produced by the *unmodified* float64-JS HDBSCAN pipeline on
 * every `__fixtures__/hdbscan/*.json` fixture.
 *
 * Later subtasks (54.7 / 54.9) assert that the tfjs front-half reproduces these
 * labels exactly (up to cluster-id permutation, consistent `-1` noise) and that
 * probability drift stays within the re-based float32 bounds. The oracle is the
 * "before" state and must be regenerated only when the fixtures themselves
 * change — never to absorb drift introduced by the migration.
 *
 * Output: `__fixtures__/hdbscan/__oracle__/hdbscan_oracle.json`
 * (a non-`.json` directory entry, so the test's `load_fixtures` readdir skips it).
 *
 * Run with: `npm run hdbscan:oracle`
 */
import fs from 'fs';
import path from 'path';

import { HDBSCAN } from '../src/clustering/hdbscan';
import type { HDBSCANParams } from '../src/clustering/types';

const FIXTURE_DIR = path.join(process.cwd(), '__fixtures__', 'hdbscan');
const ORACLE_DIR = path.join(FIXTURE_DIR, '__oracle__');
const ORACLE_PATH = path.join(ORACLE_DIR, 'hdbscan_oracle.json');

interface HdbscanFixture {
  params: {
    min_cluster_size: number;
    min_samples: number | null;
    cluster_selection_method: 'eom' | 'leaf';
    cluster_selection_epsilon: number;
    metric: 'euclidean' | 'manhattan' | 'precomputed';
  };
  tie_free: boolean;
  X?: number[][];
  distance_matrix?: number[][];
}

interface OracleEntry {
  file: string;
  tie_free: boolean;
  metric: string;
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

async function main(): Promise<void> {
  const files = fs
    .readdirSync(FIXTURE_DIR)
    .filter((f) => f.endsWith('.json'))
    .sort();

  const entries: OracleEntry[] = [];

  for (const file of files) {
    const fixture = JSON.parse(
      fs.readFileSync(path.join(FIXTURE_DIR, file), 'utf-8'),
    ) as HdbscanFixture;

    const model = new HDBSCAN(fixture_params(fixture));
    await model.fit(fit_input(fixture));

    entries.push({
      file,
      tie_free: fixture.tie_free,
      metric: fixture.params.metric,
      labels: model.labels_!,
      probabilities: model.probabilities_!,
    });
    model.dispose();
    console.log(`  captured ${file} (${entries[entries.length - 1].labels.length} points)`);
  }

  fs.mkdirSync(ORACLE_DIR, { recursive: true });
  fs.writeFileSync(ORACLE_PATH, JSON.stringify(entries, null, 2) + '\n');

  console.log(
    `\nWrote oracle for ${entries.length} fixtures to ${path.relative(process.cwd(), ORACLE_PATH)}`,
  );
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
