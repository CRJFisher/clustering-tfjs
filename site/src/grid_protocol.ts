// The main↔worker contract for the clustering grid. Imported by both the
// main-thread orchestrator (grid.ts) and the worker (grid_worker.ts) so the two
// ends can never drift. One worker runs every cell sequentially, streaming a
// result per cell.

import type { GridAlgorithmId, GridDatasetId, GridParams } from "./grid_config";

// A single 2-D dataset, generated once on the main thread and shared by every
// algorithm column that clusters it. Sent once per dataset (not once per cell) so
// the request never duplicates the same points five times.
export interface GridDatasetPayload {
  id: GridDatasetId;
  n_samples: number;
  // Row-major n_samples × 2, float32, standardized.
  data: Float32Array;
}

// One cell to compute: which dataset, which algorithm, and the exact params.
export interface GridJob {
  cell_id: string;
  dataset_id: GridDatasetId;
  algorithm_id: GridAlgorithmId;
  params: GridParams;
}

// main → worker: compute the whole grid. The worker initializes one backend, then
// fits every job in order against the matching dataset.
export interface GridRequest {
  type: "run";
  datasets: GridDatasetPayload[];
  jobs: GridJob[];
}

export type GridInbound = GridRequest;

// worker → main: how far through the matrix the worker is, so the UI can show a
// live "N / 25 computed" readout without waiting for the whole grid.
export interface GridProgress {
  type: "progress";
  completed: number;
  total: number;
}

// worker → main: one cell's labels. Streamed the instant the fit lands so the UI
// paints each scatter as it arrives.
export interface GridCellResult {
  type: "cell_result";
  cell_id: string;
  // length n_samples; -1 marks noise (HDBSCAN).
  labels: Int32Array;
  // Distinct non-noise labels — drives the per-cell "k found" annotation.
  n_clusters_found: number;
  noise_count: number;
}

// worker → main: one cell threw (an algorithm that rejected its params or wedged
// on a shape). The sibling 24 cells continue; only this cell is marked failed.
export interface GridCellError {
  type: "cell_error";
  cell_id: string;
  message: string;
}

// worker → main: the whole matrix is done. Carries the honest backend label and
// engine version that actually computed every cell.
export interface GridDone {
  type: "done";
  actual_backend: string;
  tfjs_version: string;
}

export type GridOutbound =
  | GridProgress
  | GridCellResult
  | GridCellError
  | GridDone;
