/// <reference lib="webworker" />
import * as tf from "@tensorflow/tfjs-core";
// Register the chained tensor methods (x.square(), x.sub(), x.min(), …) the
// library's algorithms call internally. tfjs-core's index registers these as
// side effects, but the production bundler tree-shakes them out of the worker
// chunk (the worker itself never calls a chained op directly), so the library's
// bundled-in calls would hit an unregistered prototype. This bare side-effect
// import pins the registration into the bundle. The benchmark worker doesn't need
// it — its affinity workload uses only functional ops.
import "@tensorflow/tfjs-core/dist/public/chained_ops/register_all_chained_ops";
import {
  Clustering,
  KMeans,
  SpectralClustering,
  AgglomerativeClustering,
  HDBSCAN,
  SOM,
} from "clustering-tfjs";
import type { GridDatasetId, GridParams } from "./grid_config";
import type {
  GridDatasetPayload,
  GridInbound,
  GridJob,
  GridOutbound,
} from "./grid_protocol";

// One worker computes the whole clustering grid: it initializes ONE backend, keeps
// the tfjs engine warm, and fits all 25 cells in sequence, streaming each cell's
// labels the instant they land. The page never freezes because every fit is off
// the main thread.

const worker_self = self as DedicatedWorkerGlobalScope;

function post(message: GridOutbound): void {
  worker_self.postMessage(message);
}

// Synchronous tensor readback inside fit_predict is unsupported on webgpu, so the
// grid never requests it (unlike the benchmark, which only awaits an affinity matrix).
// This chain prefers in-browser GPU (webgl), then wasm, then the universal cpu
// floor — every entry supports the synchronous reads the algorithms perform.
const BACKEND_CHAIN = ["webgl", "wasm", "cpu"] as const;

async function init_backend(): Promise<string> {
  let last_error: unknown;
  for (const candidate of BACKEND_CHAIN) {
    try {
      await Clustering.init({ backend: candidate });
      return tf.getBackend();
    } catch (error) {
      last_error = error;
      Clustering.reset();
    }
  }
  throw last_error instanceof Error
    ? last_error
    : new Error("No fit-predict-capable backend could be initialized.");
}

function to_rows(payload: GridDatasetPayload): number[][] {
  const rows: number[][] = [];
  for (let i = 0; i < payload.n_samples; i++) {
    rows.push([payload.data[i * 2], payload.data[i * 2 + 1]]);
  }
  return rows;
}

// SOM needs the two-call fit-then-cluster path (its raw BMU labels outnumber the
// desired clusters); every other algorithm shares the single fit_predict shape.
async function fit_labels(
  params: GridParams,
  data: number[][],
): Promise<number[]> {
  switch (params.algorithm_id) {
    case "kmeans":
      return new KMeans({
        n_clusters: params.n_clusters,
        n_init: params.n_init,
        random_state: params.random_state,
      }).fit_predict(data);
    case "spectral":
      return new SpectralClustering({
        n_clusters: params.n_clusters,
        affinity: params.affinity,
        gamma: params.gamma,
        n_neighbors: params.n_neighbors,
        n_init: params.n_init,
        random_state: params.random_state,
      }).fit_predict(data);
    case "agglomerative":
      return new AgglomerativeClustering({
        n_clusters: params.n_clusters,
        linkage: params.linkage,
      }).fit_predict(data);
    case "hdbscan":
      return new HDBSCAN({
        min_cluster_size: params.min_cluster_size,
        min_samples: params.min_samples,
      }).fit_predict(data);
    case "som": {
      const som = new SOM({
        grid_width: params.grid_width,
        grid_height: params.grid_height,
        num_epochs: params.num_epochs,
        random_state: params.random_state,
      });
      await som.fit(data);
      return som.cluster(params.n_clusters, { linkage: params.cluster_linkage });
    }
  }
}

function summarize(labels: number[]): {
  n_clusters_found: number;
  noise_count: number;
} {
  const distinct = new Set<number>();
  let noise_count = 0;
  for (const label of labels) {
    if (label < 0) noise_count += 1;
    else distinct.add(label);
  }
  return { n_clusters_found: distinct.size, noise_count };
}

// Fit one cell against the uploaded datasets, streaming its result or a per-cell
// error — one failing cell never takes down its siblings.
async function fit_cell(
  job: GridJob,
  datasets: Map<GridDatasetId, number[][]>,
): Promise<void> {
  const data = datasets.get(job.dataset_id);
  if (!data) {
    post({
      type: "cell_error",
      cell_id: job.cell_id,
      message: `No dataset payload for ${job.dataset_id}`,
    });
    return;
  }
  try {
    const labels = await fit_labels(job.params, data);
    const { n_clusters_found, noise_count } = summarize(labels);
    post({
      type: "cell_result",
      cell_id: job.cell_id,
      labels: Int32Array.from(labels),
      n_clusters_found,
      noise_count,
    });
  } catch (error) {
    post({
      type: "cell_error",
      cell_id: job.cell_id,
      message: error instanceof Error ? error.message : String(error),
    });
  }
}

async function handle_run(
  datasets: GridDatasetPayload[],
  jobs: GridJob[],
): Promise<void> {
  const cached_datasets = new Map<GridDatasetId, number[][]>();
  for (const payload of datasets) {
    cached_datasets.set(payload.id, to_rows(payload));
  }

  const total = jobs.length;
  post({ type: "progress", completed: 0, total });

  let actual_backend: string;
  try {
    actual_backend = await init_backend();
  } catch (error) {
    // Init failure is fatal for the whole grid: report every cell as failed so no
    // canvas is left in a permanent "computing" state, then stop.
    const message = error instanceof Error ? error.message : String(error);
    for (const job of jobs) {
      post({ type: "cell_error", cell_id: job.cell_id, message });
    }
    post({ type: "done", actual_backend: "none", tfjs_version: tf.version_core });
    return;
  }

  let completed = 0;
  for (const job of jobs) {
    await fit_cell(job, cached_datasets);
    completed += 1;
    post({ type: "progress", completed, total });
  }

  post({ type: "done", actual_backend, tfjs_version: tf.version_core });
}

worker_self.onmessage = (event: MessageEvent<GridInbound>) => {
  const message = event.data;
  switch (message.type) {
    case "run":
      void handle_run(message.datasets, message.jobs);
      return;
    default: {
      const unreachable: never = message.type;
      throw new Error(`Unhandled worker message: ${unreachable}`);
    }
  }
};
