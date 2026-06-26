import { make_blobs_js } from "./make_blobs_js";
import type { MakeBlobsResult } from "./make_blobs_js";
import type {
  BackendLane,
  RaceRequest,
  RaceResult,
  WorkerOutbound,
} from "./race_protocol";

// The main thread is a pure orchestrator: it generates ONE seeded dataset,
// hands an identical copy to each lane's worker, and renders results. It never
// initializes tfjs — all compute is off the main thread so the page never
// competes with the GPU for frames.

export interface RaceConfig {
  n_samples: number;
  n_features: number;
  centers: number;
  cluster_std: number;
  random_state: number;
  gamma: number;
  warmups: number;
  reps: number;
}

export const DEFAULT_RACE_CONFIG: RaceConfig = {
  // make_blobs n=2000, d=32 is the task's default headline workload: large
  // enough that the O(n²·d) affinity favors the GPU, small enough to stay well
  // under the dense-matrix memory ceiling on low-end devices.
  n_samples: 2000,
  n_features: 32,
  centers: 4,
  cluster_std: 1.5,
  random_state: 42,
  // 1/n_features mirrors the library/scikit-learn default, pinned explicitly so
  // both lanes compute the identical kernel.
  gamma: 1 / 32,
  warmups: 3,
  reps: 7,
};

export interface RaceCallbacks {
  // Fires once with the seeded dataset before any lane starts, so the UI can
  // project and draw both scatter panels OUTSIDE every timed region — the page
  // must never redraw a chart while a measured run is in flight.
  on_dataset?: (dataset: MakeBlobsResult) => void;
  on_progress?: (lane: BackendLane, phase: string, rep?: number) => void;
  on_lane_result?: (result: RaceResult) => void;
  on_lane_error?: (lane: BackendLane, message: string) => void;
}

function spawn_worker(): Worker {
  return new Worker(new URL("./race_worker.ts", import.meta.url), {
    type: "module",
  });
}

// A lane that wedges (a GPU that never flushes, a backend that hangs on init)
// would otherwise leave its promise pending forever and freeze the UI. The
// timeout terminates the worker and rejects so a single bad lane cannot hang the
// whole race.
const LANE_TIMEOUT_MS = 60_000;

function run_lane(
  lane: BackendLane,
  config: RaceConfig,
  data: Float32Array,
  callbacks: RaceCallbacks,
): Promise<RaceResult> {
  return new Promise((resolve, reject) => {
    const worker = spawn_worker();
    const timer = setTimeout(() => {
      worker.terminate();
      reject(new Error(`${lane} lane timed out after ${LANE_TIMEOUT_MS} ms`));
    }, LANE_TIMEOUT_MS);

    const settle_resolve = (result: RaceResult): void => {
      clearTimeout(timer);
      worker.terminate();
      resolve(result);
    };
    const settle_reject = (error: Error): void => {
      clearTimeout(timer);
      worker.terminate();
      reject(error);
    };

    worker.onmessage = (event: MessageEvent<WorkerOutbound>) => {
      const message = event.data;
      if (message.type === "progress") {
        callbacks.on_progress?.(lane, message.phase, message.rep);
        return;
      }
      if (message.type === "result") {
        callbacks.on_lane_result?.(message);
        settle_resolve(message);
        return;
      }
      if (message.type === "error") {
        callbacks.on_lane_error?.(lane, message.message);
        settle_reject(new Error(message.message));
        return;
      }
      const unreachable: never = message;
      settle_reject(
        new Error(`Unhandled worker message: ${JSON.stringify(unreachable)}`),
      );
    };

    worker.onerror = (event) => {
      callbacks.on_lane_error?.(lane, event.message);
      settle_reject(new Error(event.message));
    };

    // A fresh copy per worker: a single transferred buffer would detach after
    // the first post and leave the second lane with nothing.
    const request: RaceRequest = {
      type: "run",
      lane,
      n_samples: config.n_samples,
      n_features: config.n_features,
      gamma: config.gamma,
      data: data.slice(),
      warmups: config.warmups,
      reps: config.reps,
    };
    worker.postMessage(request);
  });
}

export interface RaceOutcome {
  cpu: RaceResult;
  gpu: RaceResult;
  // The backend the GPU lane actually ran (webgpu, or webgl when it fell back).
  // The headline must name this, never the requested lane, so a WebGL fallback
  // is never presented as a WebGPU win.
  gpu_backend: string;
  speedup: number;
}

export async function run_race(
  config: RaceConfig = DEFAULT_RACE_CONFIG,
  callbacks: RaceCallbacks = {},
): Promise<RaceOutcome> {
  const dataset = make_blobs_js({
    n_samples: config.n_samples,
    n_features: config.n_features,
    centers: config.centers,
    cluster_std: config.cluster_std,
    random_state: config.random_state,
  });
  callbacks.on_dataset?.(dataset);
  const { data } = dataset;

  // allSettled (not Promise.all): both lanes always run to completion or their
  // own timeout, and each terminates its own worker on settle, so one lane's
  // failure can never strand the sibling worker.
  const [cpu_settled, gpu_settled] = await Promise.allSettled([
    run_lane("cpu", config, data, callbacks),
    run_lane("webgpu", config, data, callbacks),
  ]);

  if (cpu_settled.status === "rejected") throw cpu_settled.reason;
  if (gpu_settled.status === "rejected") throw gpu_settled.reason;

  const cpu = cpu_settled.value;
  const gpu = gpu_settled.value;
  return {
    cpu,
    gpu,
    gpu_backend: gpu.actual_backend,
    speedup: cpu.median_ms / gpu.median_ms,
  };
}
