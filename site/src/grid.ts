import { make_toy_dataset, type ToyDataset } from "./make_toy_datasets";
import { GRID_CELLS, GRID_DATASETS } from "./grid_config";
import type { GridDatasetId } from "./grid_config";
import type {
  GridDatasetPayload,
  GridJob,
  GridOutbound,
} from "./grid_protocol";

// Main-thread orchestrator for the clustering grid. One long-lived worker is
// spawned the instant the runner is created (page load) and told to initialize its
// backend right away, so the honest backend label reaches the UI before any
// clustering. On `run` the orchestrator generates the five datasets once, hands
// them plus the 25 jobs to that already-warm worker, and streams the per-cell
// results back. Like the benchmark orchestrator it never initializes tfjs itself —
// all compute is off the main thread so the page never freezes while 25 cells fit.

export interface GridDatasets {
  by_id: Map<GridDatasetId, ToyDataset>;
}

// Backend lifecycle, reported once each as init resolves — independent of any
// clustering run, so the UI can label the backend the moment the page loads.
export interface GridLifecycle {
  on_backend_ready: (actual_backend: string, tfjs_version: string) => void;
  on_backend_error: (message: string) => void;
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
  // The sweep reached its terminal edge (every cell reported, or the watchdog
  // fired). The backend label was already delivered via GridLifecycle.
  on_done: () => void;
}

export interface GridRunner {
  // Kick off the 25-cell sweep on the already-warm worker. Idempotent at the UI
  // layer (the button disables), but calling twice would post a second run.
  run: (callbacks: GridCallbacks) => void;
}

// A wedged fit (an algorithm that never returns on some shape) would otherwise
// leave the grid stuck mid-stream. This terminates the worker and ends the run so
// the section can never hang forever. Backend init has its own, longer warm-up and
// is reported separately, so this covers only the fits.
const GRID_TIMEOUT_MS = 120_000;

function spawn_worker(): Worker {
  return new Worker(new URL("./grid_worker.ts", import.meta.url), {
    type: "module",
  });
}

// Spawn the worker and start backend init immediately. The returned runner's
// `run` streams the clustering results; the backend label arrives earlier, via the
// lifecycle callbacks, as soon as init resolves.
export function create_grid_runner(lifecycle: GridLifecycle): GridRunner {
  const worker = spawn_worker();
  let run_callbacks: GridCallbacks | null = null;
  let run_timer: ReturnType<typeof setTimeout> | undefined;
  // Tracks which phase a worker crash lands in, so onerror routes to the right
  // callback: a crash during init is a backend failure; during a run it ends the run.
  let running = false;

  function finish_run(): void {
    if (run_timer !== undefined) clearTimeout(run_timer);
    running = false;
    worker.terminate();
    run_callbacks?.on_done();
  }

  worker.onmessage = (event: MessageEvent<GridOutbound>) => {
    const message = event.data;
    switch (message.type) {
      case "backend_ready":
        lifecycle.on_backend_ready(message.actual_backend, message.tfjs_version);
        return;
      case "backend_error":
        lifecycle.on_backend_error(message.message);
        return;
      case "progress":
        run_callbacks?.on_progress(message.completed, message.total);
        return;
      case "cell_result":
        run_callbacks?.on_cell_result(
          message.cell_id,
          message.labels,
          message.n_clusters_found,
          message.noise_count,
        );
        return;
      case "cell_error":
        run_callbacks?.on_cell_error(message.cell_id, message.message);
        return;
      case "done":
        finish_run();
        return;
      default: {
        const unreachable: never = message;
        throw new Error(`Unhandled worker message: ${JSON.stringify(unreachable)}`);
      }
    }
  };

  worker.onerror = (event) => {
    if (running) {
      finish_run();
    } else {
      worker.terminate();
      lifecycle.on_backend_error(`worker error: ${event.message}`);
    }
  };

  // Start the backend warming up now, before the visitor asks to cluster anything.
  worker.postMessage({ type: "init" });

  return {
    run(callbacks: GridCallbacks): void {
      run_callbacks = callbacks;
      running = true;

      const by_id = new Map<GridDatasetId, ToyDataset>();
      const payloads: GridDatasetPayload[] = GRID_DATASETS.map((spec) => {
        const dataset = make_toy_dataset(spec.id);
        by_id.set(spec.id, dataset);
        return {
          id: spec.id,
          n_samples: dataset.n_samples,
          // A fresh copy per payload: structured clone would otherwise detach a
          // buffer shared with the projection the UI renders from.
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

      run_timer = setTimeout(() => finish_run(), GRID_TIMEOUT_MS);
      worker.postMessage({ type: "run", datasets: payloads, jobs });
    },
  };
}
