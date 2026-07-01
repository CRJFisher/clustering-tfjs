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

// main → worker: initialize the backend now, before any clustering. Sent the
// instant the worker is spawned (page load) so the honest backend label lands in
// the UI long before the visitor clicks to populate the grid.
export interface GridInit {
  type: "init";
}

// main → worker: compute the whole grid. The worker reuses the already-warm
// backend from the earlier init, then fits every job in order against the
// matching dataset.
export interface GridRequest {
  type: "run";
  datasets: GridDatasetPayload[];
  jobs: GridJob[];
}

export type GridInbound = GridInit | GridRequest;

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

// worker → main: the backend finished initializing. Fires in response to `init`,
// before any clustering, so the UI can show the honest backend label and engine
// version immediately. Carries the label the fits will actually run on.
export interface GridBackendReady {
  type: "backend_ready";
  actual_backend: string;
  tfjs_version: string;
}

// worker → main: every candidate backend failed to initialize. The grid cannot
// cluster; the UI marks the backend unavailable.
export interface GridBackendError {
  type: "backend_error";
  message: string;
}

// worker → main: the whole matrix is done. The backend label was already reported
// by `backend_ready`, so this only signals the sweep's terminal edge.
export interface GridDone {
  type: "done";
}

export type GridOutbound =
  | GridBackendReady
  | GridBackendError
  | GridProgress
  | GridCellResult
  | GridCellError
  | GridDone;
