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
} from "../src/utils/affinity";

import {
  degree_vector,
  normalised_laplacian,
  smallest_eigenvectors,
} from "../src/utils/laplacian";

import { KMeans } from "../src/clustering/kmeans";

/* -------------------------------- CLI args -------------------------------- */

const argv = process.argv.slice(2);
let filterRegex: RegExp | null = null;
if (argv.length >= 2 && (argv[0] === "--filter" || argv[0] === "-f")) {
  // Treat plain string as substring match, otherwise allow JS regex literal.
  const pattern = argv[1];
  filterRegex = new RegExp(pattern);
}

/* --------------------------- Helper – cosine dist ------------------------- */

function cosineDistance(a: number[], b: number[]): number {
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
  .filter((f) => (filterRegex ? filterRegex.test(f) : true))
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
  const outDir = path.join(OUT_ROOT, name);
  fs.mkdirSync(outDir, { recursive: true });

  /* ----------------------------- Load fixture ---------------------------- */
  const fixtureRaw = JSON.parse(
    fs.readFileSync(path.join(FIXTURE_DIR, file), "utf-8"),
  ) as {
    X: number[][];
    params: {
      nClusters: number;
      affinity: string;
      gamma?: number;
      nNeighbors?: number;
      randomState?: number;
    };
    labels: number[];
  };

  const { X, params } = fixtureRaw;
  const Xtensor = tf.tensor2d(X, undefined, "float32");

  /* ------------------------- 1) Affinity matrix ------------------------- */
  let A: tf.Tensor2D;
  if (typeof params.affinity === "string" && params.affinity === "rbf") {
    A = compute_rbf_affinity(Xtensor, params.gamma);
  } else if (params.affinity === "nearest_neighbors") {
    const k = params.nNeighbors ?? 10;
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
  const U_full = smallest_eigenvectors(L, params.nClusters); // n × (k+1)

  /* ------------------------ 4) Build embedding -------------------------- */
  const U = tf.slice(U_full, [0, 1], [-1, params.nClusters]) as tf.Tensor2D; // drop trivial

  const eps = 1e-10;
  const U_norm = tf.tidy(() => {
    const rowNorm = U.norm("euclidean", 1).expandDims(1);
    return U.div(rowNorm.add(eps));
  }) as tf.Tensor2D;

  /* ------------------------- 5) K-Means labels -------------------------- */
  const km = new KMeans({
    nClusters: params.nClusters,
    randomState: params.randomState,
  });
  await km.fit(U_norm);
  const labelsPred = km.labels_ as Int32Array | number[];

  /* ----------------------------- Dump artefacts ------------------------- */
  const eigArr = (await U_full.array()) as number[][];
  const embedArr = (await U_norm.array()) as number[][];

  fs.writeFileSync(path.join(outDir, "eig_sorted.json"), JSON.stringify(eigArr));
  fs.writeFileSync(path.join(outDir, "embedding.json"), JSON.stringify(embedArr));
  fs.writeFileSync(path.join(outDir, "labels.json"), JSON.stringify(Array.from(labelsPred)));

  /* --------------------------- Comparison (optional) -------------------- */
  let summary = "";
  const expectedEmbeddingPath = path.join(
    FIXTURE_DIR,
    `${name}_expected_embedding.json`,
  );
  const expectedLabelsPath = path.join(FIXTURE_DIR, `${name}_expected_labels.json`);

  let firstBadIdx: number | null = null;
  if (fs.existsSync(expectedEmbeddingPath)) {
    const expectedEmb = JSON.parse(fs.readFileSync(expectedEmbeddingPath, "utf8"));
    const tol = 1e-6;
    for (let i = 0; i < embedArr.length; i++) {
      const d = cosineDistance(embedArr[i], expectedEmb[i]);
      if (d > tol) {
        firstBadIdx = i;
        break;
      }
    }
  }

  let ariVal: number | null = null;
  if (fs.existsSync(expectedLabelsPath)) {
    const expectedLab = JSON.parse(fs.readFileSync(expectedLabelsPath, "utf8"));
    ariVal = adjustedRandIndex(Array.from(labelsPred), expectedLab);
  }

  if (firstBadIdx == null) {
    summary += "embedding: OK\n";
  } else {
    summary += `embedding: first mismatch at row ${firstBadIdx}\n`;
  }

  if (ariVal != null) {
    summary += `labels ARI = ${ariVal.toFixed(6)}\n`;
  }

  fs.writeFileSync(path.join(outDir, "summary.txt"), summary);

  console.log(
    `[${name}] ` + (firstBadIdx == null ? "embedding OK" : `embedding diff @${firstBadIdx}`),
  );

  /* ------------------------------ Cleanup ------------------------------ */
  tf.dispose([Xtensor, A, L, U_full, U, U_norm]);
}

/* ---------------------- Adjusted Rand Index helper ---------------------- */

function adjustedRandIndex(labelsA: number[], labelsB: number[]): number {
  if (labelsA.length !== labelsB.length) {
    throw new Error("Label arrays must have same length");
  }

  const n = labelsA.length;
  const labelToIndexA = new Map<number, number>();
  const labelToIndexB = new Map<number, number>();

  let nextA = 0;
  let nextB = 0;

  const contingency: number[][] = [];

  for (let i = 0; i < n; i++) {
    const a = labelsA[i];
    const b = labelsB[i];

    if (!labelToIndexA.has(a)) {
      labelToIndexA.set(a, nextA++);
      contingency.push([]);
    }
    const idxA = labelToIndexA.get(a)!;

    if (!labelToIndexB.has(b)) {
      labelToIndexB.set(b, nextB++);
      for (const row of contingency) row.push(0);
    }
    const idxB = labelToIndexB.get(b)!;

    contingency[idxA][idxB] = (contingency[idxA][idxB] || 0) + 1;
  }

  const ai = contingency.map((row) => row.reduce((s, v) => s + v, 0));
  const bj = contingency[0].map((_, j) => contingency.reduce((s, row) => s + row[j], 0));

  const comb2 = (x: number): number => (x * (x - 1)) / 2;

  let sumComb = 0;
  for (const row of contingency) {
    for (const val of row) sumComb += comb2(val);
  }

  const sumAi = ai.reduce((s, v) => s + comb2(v), 0);
  const sumBj = bj.reduce((s, v) => s + comb2(v), 0);

  const expected = (sumAi * sumBj) / comb2(n);
  const max = (sumAi + sumBj) / 2;

  if (max === expected) return 0;
  return (sumComb - expected) / (max - expected);
}

