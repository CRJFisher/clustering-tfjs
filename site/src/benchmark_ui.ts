import { run_benchmark } from "./benchmark";
import {
  BENCH_SWEEP,
  DEFAULT_BENCHMARK_CONFIG,
  N_MAX,
  N_MIN,
} from "./benchmark_sweep";
import type { BenchmarkPoint, BenchmarkSummary } from "./benchmark_sweep";
import type { SeriesId } from "./benchmark_protocol";
import { render_chart } from "./chart_canvas";
import type { ChartPoint, ChartSeries } from "./chart_canvas";
import { project_2d_pca } from "./project_2d";
import { render_scatter } from "./scatter_canvas";

// The benchmark fold's controller. It drives the pure run_benchmark orchestrator
// and translates its streamed points into incremental chart repaints — the chart
// grows a point at a time as each backend finishes a size, so the visitor watches
// the scaling law draw itself. The main thread never times compute; every plotted
// number comes straight from a worker's measured median.

// CPU muted, accelerated highlighted — the fast lane is the one to notice. Kept in
// sync with the theme accents in style.css by intent (a canvas needs concrete hex).
const SERIES_STYLE: Record<SeriesId, { label: string; color: string }> = {
  cpu: { label: "CPU", color: "#9aa4b2" },
  accelerated: { label: "Accelerated (GPU)", color: "#7c9cff" },
};

function require_el<T extends HTMLElement>(selector: string): T {
  const element = document.querySelector<T>(selector);
  if (!element) throw new Error(`Missing required element ${selector}`);
  return element;
}

function format_count(value: number): string {
  return Math.round(value).toLocaleString();
}

export interface BenchmarkController {
  run: () => void;
}

export function make_benchmark_ui(): BenchmarkController {
  const run_button = require_el<HTMLButtonElement>("#run-benchmark");
  const chart_canvas = require_el<HTMLCanvasElement>("#benchmark-chart");
  const scatter_canvas = require_el<HTMLCanvasElement>("#benchmark-scatter");
  const status = require_el<HTMLElement>("#benchmark-status");

  const tile_speedup = require_el<HTMLElement>("#tile-speedup");
  const tile_speedup_label = require_el<HTMLElement>("#tile-speedup-label");
  const tile_throughput = require_el<HTMLElement>("#tile-throughput");

  const config_cpu_backend = require_el<HTMLElement>("#config-cpu-backend");
  const config_accel_backend = require_el<HTMLElement>("#config-accel-backend");
  const config_tfjs_version = require_el<HTMLElement>("#config-tfjs-version");

  // Seed the methodology panel's fixed fields from the one config the workers
  // actually run, so the documented schedule can never drift from the code.
  require_el<HTMLElement>("#methodology-warmups").textContent = String(
    DEFAULT_BENCHMARK_CONFIG.warmups,
  );
  require_el<HTMLElement>("#methodology-reps").textContent = String(
    DEFAULT_BENCHMARK_CONFIG.reps,
  );
  require_el<HTMLElement>("#config-d").textContent = String(
    DEFAULT_BENCHMARK_CONFIG.n_features,
  );
  require_el<HTMLElement>("#config-schedule").textContent =
    `${DEFAULT_BENCHMARK_CONFIG.warmups} · ${DEFAULT_BENCHMARK_CONFIG.reps}`;
  require_el<HTMLElement>("#config-sweep").textContent =
    `${format_count(N_MIN)} → ${format_count(N_MAX)}`;

  const points: Record<SeriesId, ChartPoint[]> = { cpu: [], accelerated: [] };

  function chart_series(): ChartSeries[] {
    return (Object.keys(points) as SeriesId[]).map((series) => ({
      label: SERIES_STYLE[series].label,
      color: SERIES_STYLE[series].color,
      points: points[series],
    }));
  }

  function repaint(): void {
    render_chart(chart_canvas, chart_series(), {
      x_domain: [N_MIN, N_MAX],
      x_ticks: BENCH_SWEEP,
    });
  }

  // The representative scatter is rendered ONCE, from the first dataset generated,
  // so the visitor sees the blob structure being clustered without any redraw
  // landing inside a timed region.
  let scatter_drawn = false;

  function reset(): void {
    points.cpu = [];
    points.accelerated = [];
    tile_speedup.textContent = "—";
    tile_speedup_label.textContent = "peak speedup";
    tile_throughput.textContent = "—";
    config_cpu_backend.textContent = "—";
    config_accel_backend.textContent = "—";
    config_tfjs_version.textContent = "—";
    repaint();
  }

  function render_summary(summary: BenchmarkSummary): void {
    if (summary.peak_speedup > 0) {
      tile_speedup.textContent = `${summary.peak_speedup.toFixed(1)}×`;
      tile_speedup_label.textContent = `peak speedup · n=${format_count(summary.peak_speedup_n)}`;
      tile_throughput.textContent = `${format_count(summary.throughput_pts_per_sec)} pts/s`;
    }
  }

  async function run(): Promise<void> {
    run_button.disabled = true;
    scatter_drawn = false;
    reset();
    status.textContent = "Initializing backends…";

    const stopped: string[] = [];
    try {
      const summary = await run_benchmark({
        on_backend: (series, actual_backend, tfjs_version) => {
          const label = actual_backend.toUpperCase();
          if (series === "cpu") config_cpu_backend.textContent = label;
          else config_accel_backend.textContent = label;
          config_tfjs_version.textContent = tfjs_version;
        },
        on_dataset: (dataset, n_samples) => {
          if (scatter_drawn) return;
          scatter_drawn = true;
          const projection = project_2d_pca(
            dataset.data,
            n_samples,
            DEFAULT_BENCHMARK_CONFIG.n_features,
          );
          render_scatter(scatter_canvas, projection, dataset.labels);
        },
        on_lane_start: (series, n_samples) => {
          status.textContent = `Benchmarking ${SERIES_STYLE[series].label} at n = ${format_count(n_samples)}…`;
        },
        on_point: (point: BenchmarkPoint) => {
          points[point.series].push({ x: point.n_samples, y: point.median_ms });
          repaint();
        },
        on_lane_stopped: (series, reason) => {
          stopped.push(`${SERIES_STYLE[series].label} stopped: ${reason}`);
        },
      });
      render_summary(summary);
      const done =
        summary.peak_speedup > 0
          ? `Done — accelerated peaks at ${summary.peak_speedup.toFixed(1)}× over CPU.`
          : "Done.";
      status.textContent = stopped.length
        ? `${done} ${stopped.join(" · ")}`
        : done;
    } catch (error) {
      status.textContent =
        error instanceof Error
          ? `Benchmark failed: ${error.message}`
          : "Benchmark failed.";
    } finally {
      run_button.disabled = false;
    }
  }

  run_button.addEventListener("click", () => void run());

  // Draw the empty axes immediately so the fold reads as a chart before the first
  // run, not a blank box.
  repaint();

  return { run: () => void run() };
}
