// The main↔worker contract for the WebGPU-vs-CPU race. Imported by both the
// main-thread orchestrator (race.ts) and every worker (race_worker.ts) so the
// two ends can never drift out of sync.

// The backend a lane is asked to run. The GPU lane is requested as 'webgpu' but
// may resolve to 'webgl' when WebGPU is unavailable — `actual_backend` on the
// result reports what actually ran so the UI can relabel honestly.
export type BackendLane = "cpu" | "webgpu" | "webgl" | "wasm";

// main → worker: run the affinity bench on one backend over a shared dataset.
export interface RaceRequest {
  type: "run";
  // The backend this worker should initialize. Falls back to 'webgl' when
  // 'webgpu' is requested but the worker's realm has no navigator.gpu.
  lane: BackendLane;
  n_samples: number;
  n_features: number;
  // Passed explicitly (not left to the library's 1/n_features default) so both
  // lanes compute the byte-identical RBF kernel and any speedup is pure compute.
  gamma: number;
  // ONE seeded float32 dataset, row-major, length n_samples * n_features. The
  // orchestrator structured-clones an identical copy into every lane.
  data: Float32Array;
  warmups: number;
  reps: number;
}

// worker → main: lifecycle pings so the UI can show "initializing… / warming up…
// / timing…" without waiting for the final result.
export interface RaceProgress {
  type: "progress";
  lane: BackendLane;
  phase: "init" | "warmup" | "timed";
  rep?: number;
}

// worker → main: the steady-state measurement plus the separately-disclosed
// cold first-run, the honest backend label, and the leak self-check.
export interface RaceResult {
  type: "result";
  requested_lane: BackendLane;
  // tf.getBackend() after init — 'webgpu' or the lane it fell back to.
  actual_backend: string;
  median_ms: number;
  min_ms: number;
  max_ms: number;
  // First warmup run, shader-compile / wasm-init inclusive. Disclosed in the UI
  // but never folded into the headline multiplier.
  first_run_ms: number;
  reps_ms: number[];
  points_per_sec: number;
  n_samples: number;
  n_features: number;
  // numTensors before the timed loop (input tensor already uploaded).
  tensors_baseline: number;
  // A cheap order-insensitive signature of the affinity matrix used as a
  // cross-lane "same result" check; identical lanes should agree closely.
  result_checksum: number;
}

export interface RaceError {
  type: "error";
  requested_lane: BackendLane;
  message: string;
}

export type WorkerOutbound = RaceProgress | RaceResult | RaceError;
