// The main↔worker contract for the scaling-law benchmark. Imported by both the
// main-thread orchestrator (benchmark.ts) and every worker (benchmark_worker.ts)
// so the two ends can never drift out of sync.

// The backend a lane is asked to run. The accelerated lane is requested as
// 'webgpu' but may resolve to 'webgl'/'wasm'/'cpu' when WebGPU is unavailable —
// `actual_backend` on the ready message reports what actually ran so the UI can
// relabel honestly.
export type BackendLane = "cpu" | "webgpu" | "webgl" | "wasm";

// The two lanes the benchmark plots. 'cpu' is the universal floor; 'accelerated'
// is the GPU-first lane that falls back down the chain.
export type SeriesId = "cpu" | "accelerated";

// main → worker: initialize one backend. Sent once; the worker stays warm for
// every subsequent bench message.
export interface BenchInit {
  type: "init";
  lane: BackendLane;
}

// main → worker: time the affinity bench at one sample size over a fresh seeded
// dataset. Sent repeatedly across the sweep; the worker uploads, times, and
// disposes per message, keeping the backend warm between sizes.
export interface BenchRun {
  type: "bench";
  n_samples: number;
  n_features: number;
  // Passed explicitly (not left to the library default) so every size computes
  // the byte-identical RBF kernel and the scaling curve is pure compute.
  gamma: number;
  // ONE seeded float32 dataset, row-major, length n_samples * n_features.
  data: Float32Array;
  warmups: number;
  reps: number;
}

export type WorkerInbound = BenchInit | BenchRun;

// worker → main: the lane initialized; carries the honest backend label and the
// engine version actually loaded.
export interface BenchReady {
  type: "ready";
  actual_backend: string;
  tfjs_version: string;
}

// worker → main: one measured point on the scaling curve.
export interface BenchPoint {
  type: "point";
  n_samples: number;
  median_ms: number;
  points_per_sec: number;
  // A cheap signature of the affinity matrix, summed in the identical row order
  // on every lane, so a silent fallback to a different kernel shows as a mismatch.
  result_checksum: number;
}

// worker → main: this lane failed (init failure, or an OOM at a large n). The
// orchestrator stops extending the lane rather than aborting the whole sweep.
export interface BenchError {
  type: "error";
  message: string;
}

export type WorkerOutbound = BenchReady | BenchPoint | BenchError;
