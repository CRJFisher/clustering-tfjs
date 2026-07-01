import { make_blobs_js } from "./make_blobs_js";
import type { MakeBlobsResult } from "./make_blobs_js";
import type {
  BackendLane,
  BenchRun,
  SeriesId,
  WorkerOutbound,
} from "./benchmark_protocol";
import {
  BENCH_SWEEP,
  DEFAULT_BENCHMARK_CONFIG,
  summarize,
} from "./benchmark_sweep";
import type {
  BenchmarkConfig,
  BenchmarkPoint,
  BenchmarkSummary,
} from "./benchmark_sweep";

// The main thread is a pure orchestrator: it generates ONE seeded dataset per
// sample size, hands an identical copy to each lane's warm worker, and streams the
// measured points back to the chart. It never initializes tfjs — all compute is
// off the main thread so the page never competes with the GPU for frames.

// The two lanes the chart plots, each backed by its own warm worker. 'accelerated'
// is requested as webgpu and falls back down the chain inside the worker.
const LANES: { series: SeriesId; lane: BackendLane }[] = [
  { series: "cpu", lane: "cpu" },
  { series: "accelerated", lane: "webgpu" },
];

// A lane that wedges on init or on a single size (a GPU that never flushes, a
// backend that hangs) would otherwise leave its promise pending forever. The
// timeout rejects so the orchestrator can retire that lane and keep going. This is
// a hard safety net; the adaptive budget below normally stops a slow lane long
// before a size could approach it.
const INIT_TIMEOUT_MS = 60_000;
const BENCH_TIMEOUT_MS = 60_000;

// The wall-clock a single size may cost a lane before it is stopped. Each size runs
// warmups + reps computations whose per-run cost grows ~O(n²), so the CPU lane's
// top sizes would otherwise take minutes and trip the timeout. Before benching a
// size, the orchestrator projects its cost from the lane's previous measured median
// and stops the lane — with an honest "too slow" reason — rather than spending the
// budget and plotting a number no visitor waits for. The accelerated lane stays far
// under this and sweeps the full range, so the divergence still draws itself.
const SIZE_BUDGET_MS = 10_000;

// Project a size's total cost for a lane from its previous measured single-run
// median, scaled by the O(n²) growth in n and multiplied by the runs-per-size.
function project_size_ms(
  prev_median_ms: number,
  prev_n: number,
  n: number,
  runs_per_size: number,
): number {
  const growth = (n / prev_n) ** 2;
  return prev_median_ms * growth * runs_per_size;
}

export interface BenchmarkCallbacks {
  // Fires once per size with the seeded dataset before its lanes run, so the UI
  // can draw the representative scatter OUTSIDE every timed region.
  on_dataset?: (dataset: MakeBlobsResult, n_samples: number) => void;
  // The lane is initializing / about to time this size — drives the status line.
  on_lane_start?: (series: SeriesId, n_samples: number) => void;
  // One measured point landed; the chart extends this lane's line immediately.
  on_point?: (point: BenchmarkPoint) => void;
  // This lane stopped early (init failure or OOM at a large size). The other lane
  // and the already-plotted points are kept.
  on_lane_stopped?: (series: SeriesId, reason: string) => void;
  // The honest backend label + engine version resolved for a lane (e.g.
  // accelerated → 'webgl').
  on_backend?: (
    series: SeriesId,
    actual_backend: string,
    tfjs_version: string,
  ) => void;
}

// Wraps one persistent worker into awaitable init()/bench() calls. Only ever one
// message is in flight per worker (the orchestrator runs sizes strictly in
// sequence), so a single pending settler is enough.
interface LaneWorker {
  series: SeriesId;
  init(): Promise<{ actual_backend: string; tfjs_version: string }>;
  bench(request: BenchRun): Promise<{ median_ms: number; points_per_sec: number }>;
  terminate(): void;
}

function make_lane_worker(series: SeriesId, lane: BackendLane): LaneWorker {
  const worker = new Worker(
    new URL("./benchmark_worker.ts", import.meta.url),
    { type: "module" },
  );

  let settle:
    | { resolve: (message: WorkerOutbound) => void; reject: (error: Error) => void }
    | null = null;

  worker.onmessage = (event: MessageEvent<WorkerOutbound>) => {
    const current = settle;
    settle = null;
    current?.resolve(event.data);
  };
  worker.onerror = (event) => {
    const current = settle;
    settle = null;
    current?.reject(new Error(event.message));
  };

  // Post a message and await the worker's single next outbound message, with a
  // timeout that rejects (and is the caller's signal to terminate the lane).
  function request(
    message: BenchRun | { type: "init"; lane: BackendLane },
    timeout_ms: number,
  ): Promise<WorkerOutbound> {
    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => {
        settle = null;
        reject(new Error(`${series} lane timed out after ${timeout_ms} ms`));
      }, timeout_ms);
      settle = {
        resolve: (m) => {
          clearTimeout(timer);
          resolve(m);
        },
        reject: (e) => {
          clearTimeout(timer);
          reject(e);
        },
      };
      worker.postMessage(message);
    });
  }

  return {
    series,
    async init(): Promise<{ actual_backend: string; tfjs_version: string }> {
      const message = await request({ type: "init", lane }, INIT_TIMEOUT_MS);
      if (message.type === "ready") {
        return {
          actual_backend: message.actual_backend,
          tfjs_version: message.tfjs_version,
        };
      }
      if (message.type === "error") throw new Error(message.message);
      throw new Error(`Unexpected ${message.type} during init`);
    },
    async bench(req): Promise<{ median_ms: number; points_per_sec: number }> {
      const message = await request(req, BENCH_TIMEOUT_MS);
      if (message.type === "point") {
        return {
          median_ms: message.median_ms,
          points_per_sec: message.points_per_sec,
        };
      }
      if (message.type === "error") throw new Error(message.message);
      throw new Error(`Unexpected ${message.type} during bench`);
    },
    terminate: () => worker.terminate(),
  };
}

// Run the full sweep. Each size generates one dataset; both lanes time it in
// sequence (cpu then accelerated — never concurrently, so the numbers are clean),
// and every point streams to the chart as it lands. A lane that fails a size is
// retired (its line stops) while the other lane finishes the remaining sizes.
export async function run_benchmark(
  callbacks: BenchmarkCallbacks = {},
  config: BenchmarkConfig = DEFAULT_BENCHMARK_CONFIG,
): Promise<BenchmarkSummary> {
  const workers = LANES.map(({ series, lane }) =>
    make_lane_worker(series, lane),
  );

  // Per-series medians keyed by size, used to compute the closing summary.
  const cpu_ms = new Map<number, number>();
  const accel = new Map<number, { ms: number; pts_per_sec: number }>();
  // A retired lane's worker is terminated; skip it for the rest of the sweep.
  const retired = new Set<SeriesId>();
  // The lane's most recent measured point, used to project the next size's cost
  // against the budget before committing to it.
  const last_point = new Map<SeriesId, { n: number; median_ms: number }>();
  const runs_per_size = config.warmups + config.reps;

  try {
    for (const worker of workers) {
      try {
        const { actual_backend, tfjs_version } = await worker.init();
        callbacks.on_backend?.(worker.series, actual_backend, tfjs_version);
      } catch (error) {
        retired.add(worker.series);
        worker.terminate();
        callbacks.on_lane_stopped?.(
          worker.series,
          error instanceof Error ? error.message : String(error),
        );
      }
    }

    for (const n_samples of BENCH_SWEEP) {
      if (retired.size === workers.length) break;

      const dataset = make_blobs_js({
        n_samples,
        n_features: config.n_features,
        centers: config.centers,
        cluster_std: config.cluster_std,
        random_state: config.random_state,
      });
      callbacks.on_dataset?.(dataset, n_samples);

      for (const worker of workers) {
        if (retired.has(worker.series)) continue;

        // Stop a lane before attempting a size projected to blow the per-size
        // budget, rather than spending minutes on it and tripping the timeout.
        const prev = last_point.get(worker.series);
        if (prev) {
          const projected = project_size_ms(
            prev.median_ms,
            prev.n,
            n_samples,
            runs_per_size,
          );
          if (projected > SIZE_BUDGET_MS) {
            retired.add(worker.series);
            worker.terminate();
            callbacks.on_lane_stopped?.(
              worker.series,
              `too slow past n=${prev.n.toLocaleString()} ` +
                `(next size projected ~${Math.round(projected / 1000)}s, ` +
                `over the ${SIZE_BUDGET_MS / 1000}s per-size budget)`,
            );
            continue;
          }
        }

        callbacks.on_lane_start?.(worker.series, n_samples);
        try {
          const point = await worker.bench({
            type: "bench",
            n_samples,
            n_features: config.n_features,
            gamma: config.gamma,
            // A fresh copy per worker: a transferred buffer would detach after the
            // first post and leave the next lane with nothing.
            data: dataset.data.slice(),
            warmups: config.warmups,
            reps: config.reps,
          });
          if (worker.series === "cpu") cpu_ms.set(n_samples, point.median_ms);
          else {
            accel.set(n_samples, {
              ms: point.median_ms,
              pts_per_sec: point.points_per_sec,
            });
          }
          last_point.set(worker.series, {
            n: n_samples,
            median_ms: point.median_ms,
          });
          callbacks.on_point?.({
            series: worker.series,
            n_samples,
            median_ms: point.median_ms,
            points_per_sec: point.points_per_sec,
          });
        } catch (error) {
          retired.add(worker.series);
          worker.terminate();
          callbacks.on_lane_stopped?.(
            worker.series,
            error instanceof Error ? error.message : String(error),
          );
        }
      }
    }
  } finally {
    for (const worker of workers) {
      if (!retired.has(worker.series)) worker.terminate();
    }
  }

  return summarize(cpu_ms, accel);
}
