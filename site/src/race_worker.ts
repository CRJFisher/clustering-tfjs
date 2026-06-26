/// <reference lib="webworker" />
import * as tf from "@tensorflow/tfjs-core";
import { Clustering } from "clustering-tfjs";
import { run_affinity_bench } from "./bench_harness";
import type {
  BackendLane,
  RaceRequest,
  WorkerOutbound,
} from "./race_protocol";

// One worker per backend. tfjs keeps a single global engine per JS realm, so
// each worker is the only place its backend can be active — this isolation is
// what lets WebGPU and CPU run concurrently. The worker imports tfjs-core
// directly (for tensor upload + tf.memory) AND clustering-tfjs (for the library
// loader + the published affinity workload); Vite dedupes tfjs-core so both
// resolve to the one engine Clustering.init configures.

const worker_self = self as DedicatedWorkerGlobalScope;

function post(message: WorkerOutbound): void {
  worker_self.postMessage(message);
}

function has_webgpu(): boolean {
  const nav: Navigator & { gpu?: unknown } = navigator;
  return nav.gpu != null;
}

// Establish the lane and return tf.getBackend() — the honest label of what
// actually runs. A 'webgpu' request degrades to 'webgl' when the realm lacks
// navigator.gpu or the backend fails to initialize, so the UI never claims a
// WebGPU win it did not measure.
async function init_lane(lane: BackendLane): Promise<string> {
  if (lane === "webgpu" && !has_webgpu()) {
    await Clustering.init({ backend: "webgl" });
    return tf.getBackend();
  }

  try {
    await Clustering.init({ backend: lane });
  } catch (error) {
    if (lane === "webgpu") {
      Clustering.reset();
      await Clustering.init({ backend: "webgl" });
      return tf.getBackend();
    }
    throw error;
  }

  return tf.getBackend();
}

worker_self.onmessage = async (event: MessageEvent<RaceRequest>) => {
  const request = event.data;
  if (request.type !== "run") return;

  // Hoisted so the finally can release it even when the bench throws (e.g. a
  // backend error mid-run) — otherwise a failed run would strand the input
  // tensor if the worker is ever reused for a second race.
  let x: tf.Tensor2D | undefined;
  try {
    post({ type: "progress", lane: request.lane, phase: "init" });
    const actual_backend = await init_lane(request.lane);

    // Upload the identical bytes ONCE, outside every timed region, as float32 on
    // this engine. Excluding the host→device transfer from the timed bracket on
    // both lanes is what makes the comparison measure compute, not upload.
    x = tf.tensor2d(
      request.data,
      [request.n_samples, request.n_features],
      "float32",
    );

    const result = await run_affinity_bench({
      x,
      gamma: request.gamma,
      warmups: request.warmups,
      reps: request.reps,
      on_phase: (phase, rep) =>
        post({ type: "progress", lane: request.lane, phase, rep }),
    });

    post({
      type: "result",
      requested_lane: request.lane,
      actual_backend,
      median_ms: result.median_ms,
      min_ms: result.min_ms,
      max_ms: result.max_ms,
      first_run_ms: result.first_run_ms,
      reps_ms: result.reps_ms,
      points_per_sec: request.n_samples / (result.median_ms / 1000),
      n_samples: request.n_samples,
      n_features: request.n_features,
      tensors_baseline: result.tensors_baseline,
      result_checksum: result.result_checksum,
    });
  } catch (error) {
    post({
      type: "error",
      requested_lane: request.lane,
      message: error instanceof Error ? error.message : String(error),
    });
  } finally {
    x?.dispose();
  }
};
