/// <reference lib="webworker" />
import * as tf from "@tensorflow/tfjs-core";
// Register the chained tensor methods (x.square(), x.sub(), …) the library's
// affinity construction calls internally. tfjs-core registers these as a side
// effect of its index, but the bundler tree-shakes them out of this worker chunk
// (the worker never calls a chained op directly), so the library's bundled-in
// calls would hit an unregistered prototype. This bare side-effect import pins the
// registration into the bundle.
import "@tensorflow/tfjs-core/dist/public/chained_ops/register_all_chained_ops";
import { init_lane } from "./backend_init";
import { run_affinity_bench } from "./bench_harness";
import type { WorkerInbound, WorkerOutbound } from "./benchmark_protocol";

// One long-lived worker per backend. It initializes its lane once (on the `init`
// message) and then handles every `bench` message over the same warm engine, so
// the sweep across sample sizes pays the backend-init / shader-compile cost only
// once. The worker imports tfjs-core directly (for tensor upload + tf.memory) AND
// clustering-tfjs (via init_lane + the published affinity workload); Vite dedupes
// tfjs-core so both resolve to the one engine Clustering.init configures.

const worker_self = self as DedicatedWorkerGlobalScope;

function post(message: WorkerOutbound): void {
  worker_self.postMessage(message);
}

worker_self.onmessage = async (event: MessageEvent<WorkerInbound>) => {
  const message = event.data;

  if (message.type === "init") {
    try {
      const actual_backend = await init_lane(message.lane);
      post({ type: "ready", actual_backend, tfjs_version: tf.version_core });
    } catch (error) {
      post({
        type: "error",
        message: error instanceof Error ? error.message : String(error),
      });
    }
    return;
  }

  // A bench message. Hoist the input tensor so the finally can release it even
  // when the bench throws (a backend error or an OOM at a large n) — otherwise a
  // failed run would strand it on the warm engine before the next size.
  let x: tf.Tensor2D | undefined;
  try {
    // Upload the bytes ONCE, outside the timed region, as float32 on this engine.
    // Excluding host→device transfer from the bracket keeps the curve pure compute.
    x = tf.tensor2d(
      message.data,
      [message.n_samples, message.n_features],
      "float32",
    );

    const result = await run_affinity_bench({
      x,
      gamma: message.gamma,
      warmups: message.warmups,
      reps: message.reps,
    });

    post({
      type: "point",
      n_samples: message.n_samples,
      median_ms: result.median_ms,
      points_per_sec: message.n_samples / (result.median_ms / 1000),
      result_checksum: result.result_checksum,
    });
  } catch (error) {
    post({
      type: "error",
      message: error instanceof Error ? error.message : String(error),
    });
  } finally {
    x?.dispose();
  }
};
