import * as tf from "@tensorflow/tfjs-core";
import { Clustering } from "clustering-tfjs";
import type { BackendLane } from "./benchmark_protocol";

// Backend initialization shared by every benchmark worker. tfjs keeps a single
// global engine per JS realm, so each worker is the only place its backend can be
// active — this isolation is what lets the accelerated and CPU lanes own separate
// engines. The honest label is whatever actually initialized, so the UI never
// claims a backend it did not run.

function has_webgpu(): boolean {
  const nav: Navigator & { gpu?: unknown } = navigator;
  return nav.gpu != null;
}

// The accelerated lane walks this chain in order until one backend initializes,
// so a browser without WebGPU (Firefox-Linux/Android, older Safari) still gets a
// real GPU lane (webgl), then wasm, then the universal cpu floor — the lane is
// never left empty or broken.
const GPU_FALLBACK_CHAIN: BackendLane[] = ["webgpu", "webgl", "wasm", "cpu"];

// Establish the lane and return tf.getBackend() — the honest label of what
// actually runs. The accelerated lane degrades down GPU_FALLBACK_CHAIN whenever a
// backend is missing or fails to initialize. The cpu lane has nothing to fall
// back to and is the floor.
export async function init_lane(lane: BackendLane): Promise<string> {
  if (lane !== "webgpu") {
    await Clustering.init({ backend: lane });
    return tf.getBackend();
  }

  // Skip the webgpu attempt entirely when the realm has no navigator.gpu —
  // importing the heavy backend package only to fail wastes a network fetch.
  const chain = has_webgpu()
    ? GPU_FALLBACK_CHAIN
    : GPU_FALLBACK_CHAIN.filter((backend) => backend !== "webgpu");

  let last_error: unknown;
  for (const candidate of chain) {
    try {
      await Clustering.init({ backend: candidate });
      return tf.getBackend();
    } catch (error) {
      last_error = error;
      // Clear the failed backend's partial singleton state before the next try.
      Clustering.reset();
    }
  }
  throw last_error instanceof Error
    ? last_error
    : new Error("No accelerated-lane backend could be initialized.");
}
