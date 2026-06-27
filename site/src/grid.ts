import { make_toy_dataset, type ToyDataset } from "./make_toy_datasets";
import { GRID_CELLS, GRID_DATASETS } from "./grid_config";
import type { GridDatasetId } from "./grid_config";
import type {
  GridDatasetPayload,
  GridJob,
  GridOutbound,
} from "./grid_protocol";

// Main-thread orchestrator for the parity grid: it generates the five datasets
// once, hands them plus the 25 jobs to one worker, and streams the worker's
// per-cell results back to the UI. Like the race orchestrator it never
// initializes tfjs — all compute is off the main thread so the page never freezes
// while 25 cells fit.

export interface GridDatasets {
  by_id: Map<GridDatasetId, ToyDataset>;
}

export interface GridCallbacks {
  // Fires once with the generated datasets before any fit, so the UI can build a
  // 2-D projection per cell ready to render the instant labels arrive.
  on_datasets: (datasets: GridDatasets) => void;
  on_progress: (completed: number, total: number) => void;
  on_cell_result: (
    cell_id: string,
    labels: Int32Array,
    n_clusters_found: number,
    noise_count: number,
  ) => void;
  on_cell_error: (cell_id: string, message: string) => void;
  on_done: (actual_backend: string, tfjs_version: string) => void;
}

// A wedged worker (a backend that hangs on init, an algorithm that never returns)
// would otherwise leave the grid stuck mid-stream. This terminates it and reports
// failure so the section can never hang forever.
const GRID_TIMEOUT_MS = 120_000;

function spawn_worker(): Worker {
  return new Worker(new URL("./grid_worker.ts", import.meta.url), {
    type: "module",
  });
}

export function run_grid(callbacks: GridCallbacks): void {
  const by_id = new Map<GridDatasetId, ToyDataset>();
  const payloads: GridDatasetPayload[] = GRID_DATASETS.map((spec) => {
    const dataset = make_toy_dataset(spec.id);
    by_id.set(spec.id, dataset);
    return {
      id: spec.id,
      n_samples: dataset.n_samples,
      // A fresh copy per payload: structured clone would otherwise detach a buffer
      // shared with the projection the UI renders from.
      data: dataset.data.slice(),
    };
  });
  callbacks.on_datasets({ by_id });

  const jobs: GridJob[] = GRID_CELLS.map((cell) => ({
    cell_id: cell.cell_id,
    dataset_id: cell.dataset_id,
    algorithm_id: cell.algorithm_id,
    params: cell.params,
  }));

  const worker = spawn_worker();
  const timer = setTimeout(() => {
    worker.terminate();
    callbacks.on_done("timed out", "");
  }, GRID_TIMEOUT_MS);

  worker.onmessage = (event: MessageEvent<GridOutbound>) => {
    const message = event.data;
    switch (message.type) {
      case "progress":
        callbacks.on_progress(message.completed, message.total);
        return;
      case "cell_result":
        callbacks.on_cell_result(
          message.cell_id,
          message.labels,
          message.n_clusters_found,
          message.noise_count,
        );
        return;
      case "cell_error":
        callbacks.on_cell_error(message.cell_id, message.message);
        return;
      case "done":
        clearTimeout(timer);
        worker.terminate();
        callbacks.on_done(message.actual_backend, message.tfjs_version);
        return;
      default: {
        const unreachable: never = message;
        throw new Error(`Unhandled worker message: ${JSON.stringify(unreachable)}`);
      }
    }
  };

  worker.onerror = (event) => {
    clearTimeout(timer);
    worker.terminate();
    callbacks.on_done(`worker error: ${event.message}`, "");
  };

  worker.postMessage({ type: "run", datasets: payloads, jobs });
}
