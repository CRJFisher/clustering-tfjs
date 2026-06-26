import { make_blobs_js } from "./make_blobs_js";
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
  on_progress?: (lane: BackendLane, phase: string, rep?: number) => void;
  on_lane_result?: (result: RaceResult) => void;
  on_lane_error?: (lane: BackendLane, message: string) => void;
}

function spawn_worker(): Worker {
  return new Worker(new URL("./race_worker.ts", import.meta.url), {
    type: "module",
  });
}

function run_lane(
  lane: BackendLane,
  config: RaceConfig,
  data: Float32Array,
  callbacks: RaceCallbacks,
): Promise<RaceResult> {
  return new Promise((resolve, reject) => {
    const worker = spawn_worker();

    worker.onmessage = (event: MessageEvent<WorkerOutbound>) => {
      const message = event.data;
      if (message.type === "progress") {
        callbacks.on_progress?.(lane, message.phase, message.rep);
        return;
      }
      if (message.type === "result") {
        callbacks.on_lane_result?.(message);
        worker.terminate();
        resolve(message);
        return;
      }
      callbacks.on_lane_error?.(lane, message.message);
      worker.terminate();
      reject(new Error(message.message));
    };

    worker.onerror = (event) => {
      callbacks.on_lane_error?.(lane, event.message);
      worker.terminate();
      reject(new Error(event.message));
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
  speedup: number;
}

export async function run_race(
  config: RaceConfig = DEFAULT_RACE_CONFIG,
  callbacks: RaceCallbacks = {},
): Promise<RaceOutcome> {
  const { data } = make_blobs_js({
    n_samples: config.n_samples,
    n_features: config.n_features,
    centers: config.centers,
    cluster_std: config.cluster_std,
    random_state: config.random_state,
  });

  const [cpu, gpu] = await Promise.all([
    run_lane("cpu", config, data, callbacks),
    run_lane("webgpu", config, data, callbacks),
  ]);

  return { cpu, gpu, speedup: cpu.median_ms / gpu.median_ms };
}
