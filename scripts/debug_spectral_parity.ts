#!/usr/bin/env ts-node

/* -------------------------------------------------------------------------- */
/*  Debug helper – compares our SpectralClustering pipeline to reference data  */
/* -------------------------------------------------------------------------- */

/*
 * The script iterates over JSON fixtures located in `test/fixtures/spectral/`.
 * For every fixture it re-runs the internal spectral clustering pipeline
 * (affinity → Laplacian → eigenvectors → embedding → k-means) **mirroring**
 * the implementation in `src/clustering/spectral.ts` while tapping into the
 * intermediate tensors that are otherwise discarded.
 *
 * The following artefacts are written to `tmp/debug/<fixture-name>/`:
 *   • eig_sorted.json      – (n, k+1) sorted & sign-fixed eigenvectors  
 *   • embedding.json       – (n, k)   row-normalised embedding matrix  
 *   • labels.json          – (n)      final cluster labels (Int32)
 *   • summary.txt          – human readable delta vs reference, if present
 *
 * A terse console summary is printed after each fixture.
 */

import fs from "fs";
import path from "path";

import * as tf from "@tensorflow/tfjs-node";

import {
  compute_rbf_affinity,
  compute_knn_affinity,
} from "../src/graph/affinity";

import {
  degree_vector,
  normalised_laplacian,
  smallest_eigenvectors,
} from "../src/graph/laplacian";

import { KMeans } from "../src/clustering/kmeans";

/* -------------------------------- CLI args -------------------------------- */

const argv = process.argv.slice(2);
let filter_regex: RegExp | null = null;
if (argv.length >= 2 && (argv[0] === "--filter" || argv[0] === "-f")) {
  // Treat plain string as substring match, otherwise allow JS regex literal.
  const pattern = argv[1];
  filter_regex = new RegExp(pattern);
}

/* --------------------------- Helper – cosine dist ------------------------- */

function cosine_distance(a: number[], b: number[]): number {
  let dot = 0;
  let na = 0;
  let nb = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    na += a[i] * a[i];
    nb += b[i] * b[i];
  }
  const denom = Math.sqrt(na) * Math.sqrt(nb);
  return denom === 0 ? 0 : 1 - dot / denom;
}

/* ------------------------ Fixture discovery & loop ------------------------ */

const FIXTURE_DIR = path.resolve(__dirname, "../test/fixtures/spectral");
if (!fs.existsSync(FIXTURE_DIR)) {
  console.error("[debug] spectral fixture directory not found:", FIXTURE_DIR);
  process.exit(1);
}

const files = fs
  .readdirSync(FIXTURE_DIR)
  .filter((f) => f.endsWith(".json"))
  .filter((f) => (filter_regex ? filter_regex.test(f) : true))
  .sort();

if (files.length === 0) {
  console.warn("[debug] No fixtures matched – exiting.");
  process.exit(0);
}

const OUT_ROOT = path.resolve(process.cwd(), "tmp/debug");
fs.mkdirSync(OUT_ROOT, { recursive: true });

/* -------------------------------- Main loop ------------------------------ */

for (const file of files) {
  const name = path.basename(file, ".json");
  const out_dir = path.join(OUT_ROOT, name);
  fs.mkdirSync(out_dir, { recursive: true });

  /* ----------------------------- Load fixture ---------------------------- */
  const fixture_raw = JSON.parse(
    fs.readFileSync(path.join(FIXTURE_DIR, file), "utf-8"),
  ) as {
    X: number[][];
    params: {
      n_clusters: number;
      affinity: string;
      gamma?: number;
      n_neighbors?: number;
      random_state?: number;
    };
    labels: number[];
  };

  const { X, params } = fixture_raw;
  const Xtensor = tf.tensor2d(X, undefined, "float32");

  /* ------------------------- 1) Affinity matrix ------------------------- */
  let A: tf.Tensor2D;
  if (typeof params.affinity === "string" && params.affinity === "rbf") {
    A = compute_rbf_affinity(Xtensor, params.gamma);
  } else if (params.affinity === "nearest_neighbors") {
    const k = params.n_neighbors ?? 10;
    A = compute_knn_affinity(Xtensor, k);
  } else if (params.affinity === "precomputed") {
    // Fixture encodes X directly as affinity when precomputed.
    A = Xtensor.clone();
  } else {
    console.warn(`Unsupported affinity '${params.affinity}' – skipping.`);
    Xtensor.dispose();
    continue;
  }

  /* ------------------------- 2) Laplacian ------------------------------ */
  const L = normalised_laplacian(A);

  /* --------------------- 3) Smallest eigenvectors ----------------------- */
  const U_full = smallest_eigenvectors(L, params.n_clusters); // n × (k+1)

  /* ------------------------ 4) Build embedding -------------------------- */
  const U = tf.slice(U_full, [0, 1], [-1, params.n_clusters]) as tf.Tensor2D; // drop trivial

  const eps = 1e-10;
  const U_norm = tf.tidy(() => {
    const row_norm = U.norm("euclidean", 1).expandDims(1);
    return U.div(row_norm.add(eps));
  }) as tf.Tensor2D;

  /* ------------------------- 5) K-Means labels -------------------------- */
  const km = new KMeans({
    n_clusters: params.n_clusters,
    random_state: params.random_state,
  });
  await km.fit(U_norm);
  const labels_pred = km.labels_ as Int32Array | number[];

  /* ----------------------------- Dump artefacts ------------------------- */
  const eig_arr = (await U_full.array()) as number[][];
  const embed_arr = (await U_norm.array()) as number[][];

  fs.writeFileSync(path.join(out_dir, "eig_sorted.json"), JSON.stringify(eig_arr));
  fs.writeFileSync(path.join(out_dir, "embedding.json"), JSON.stringify(embed_arr));
  fs.writeFileSync(path.join(out_dir, "labels.json"), JSON.stringify(Array.from(labels_pred)));

  /* --------------------------- Comparison (optional) -------------------- */
  let summary = "";
  const expected_embedding_path = path.join(
    FIXTURE_DIR,
    `${name}_expected_embedding.json`,
  );
  const expected_labels_path = path.join(FIXTURE_DIR, `${name}_expected_labels.json`);

  let first_bad_idx: number | null = null;
  if (fs.existsSync(expected_embedding_path)) {
    const expected_emb = JSON.parse(fs.readFileSync(expected_embedding_path, "utf8"));
    const tol = 1e-6;
    for (let i = 0; i < embed_arr.length; i++) {
      const d = cosine_distance(embed_arr[i], expected_emb[i]);
      if (d > tol) {
        first_bad_idx = i;
        break;
      }
    }
  }

  let ari_val: number | null = null;
  if (fs.existsSync(expected_labels_path)) {
    const expected_lab = JSON.parse(fs.readFileSync(expected_labels_path, "utf8"));
    ari_val = adjusted_rand_index(Array.from(labels_pred), expected_lab);
  }

  if (first_bad_idx == null) {
    summary += "embedding: OK\n";
  } else {
    summary += `embedding: first mismatch at row ${first_bad_idx}\n`;
  }

  if (ari_val != null) {
    summary += `labels ARI = ${ari_val.toFixed(6)}\n`;
  }

  fs.writeFileSync(path.join(out_dir, "summary.txt"), summary);

  console.log(
    `[${name}] ` + (first_bad_idx == null ? "embedding OK" : `embedding diff @${first_bad_idx}`),
  );

  /* ------------------------------ Cleanup ------------------------------ */
  tf.dispose([Xtensor, A, L, U_full, U, U_norm]);
}

/* ---------------------- Adjusted Rand Index helper ---------------------- */

function adjusted_rand_index(labels_a: number[], labels_b: number[]): number {
  if (labels_a.length !== labels_b.length) {
    throw new Error("Label arrays must have same length");
  }

  const n = labels_a.length;
  const label_to_index_a = new Map<number, number>();
  const label_to_index_b = new Map<number, number>();

  let next_a = 0;
  let next_b = 0;

  const contingency: number[][] = [];

  for (let i = 0; i < n; i++) {
    const a = labels_a[i];
    const b = labels_b[i];

    if (!label_to_index_a.has(a)) {
      label_to_index_a.set(a, next_a++);
      contingency.push([]);
    }
    const idx_a = label_to_index_a.get(a)!;

    if (!label_to_index_b.has(b)) {
      label_to_index_b.set(b, next_b++);
      for (const row of contingency) row.push(0);
    }
    const idx_b = label_to_index_b.get(b)!;

    contingency[idx_a][idx_b] = (contingency[idx_a][idx_b] || 0) + 1;
  }

  const ai = contingency.map((row) => row.reduce((s, v) => s + v, 0));
  const bj = contingency[0].map((_, j) => contingency.reduce((s, row) => s + row[j], 0));

  const comb2 = (x: number): number => (x * (x - 1)) / 2;

  let sum_comb = 0;
  for (const row of contingency) {
    for (const val of row) sum_comb += comb2(val);
  }

  const sum_ai = ai.reduce((s, v) => s + comb2(v), 0);
  const sum_bj = bj.reduce((s, v) => s + comb2(v), 0);

  const expected = (sum_ai * sum_bj) / comb2(n);
  const max = (sum_ai + sum_bj) / 2;

  if (max === expected) return 0;
  return (sum_comb - expected) / (max - expected);
}

