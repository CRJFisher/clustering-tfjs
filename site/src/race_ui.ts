import { run_race, DEFAULT_RACE_CONFIG } from "./race";
import type { RaceOutcome } from "./race";
import type { BackendLane, RaceResult } from "./race_protocol";
import type { MakeBlobsResult } from "./make_blobs_js";
import { project_2d_pca } from "./project_2d";
import { render_scatter } from "./scatter_canvas";
import { make_stopwatch } from "./stopwatch";

// The whole race fold. The controller is a pure orchestrator over the existing
// `run_race` harness: it translates the harness's callbacks into cheap DOM
// updates (timers, bars, tiles) and renders the scatter ONCE, before timing.
// Every published number is read straight from the worker's `RaceResult`; the
// main thread never times compute, so it can never present its own wall-clock as
// a measured figure.

// Both lanes emit `warmups + reps` progress events after init; the racing bar
// advances one step per event toward this total.
const EXPECTED_STEPS = DEFAULT_RACE_CONFIG.warmups + DEFAULT_RACE_CONFIG.reps;

// Matching float32 RBF kernels sum to within several significant figures despite
// rounding, so a relative gap this small means "same computation," while a gross
// divergence means one lane silently fell back or computed a different kernel.
const RESULT_MATCH_REL_TOL = 1e-3;

function require_el<T extends HTMLElement>(selector: string): T {
  const element = document.querySelector<T>(selector);
  if (!element) throw new Error(`Missing required element ${selector}`);
  return element;
}

function format_ms(ms: number): string {
  return `${ms.toFixed(1)} ms`;
}

function format_count(value: number): string {
  return Math.round(value).toLocaleString();
}

// Symmetric denominator (never divide by a single lane's checksum, which could
// be ~0): the gap is relative to the larger magnitude, with an epsilon floor.
function results_agree(cpu_checksum: number, gpu_checksum: number): boolean {
  const denom = Math.max(
    Math.abs(cpu_checksum),
    Math.abs(gpu_checksum),
    1e-12,
  );
  return Math.abs(cpu_checksum - gpu_checksum) / denom <= RESULT_MATCH_REL_TOL;
}

interface LaneView {
  begin(): void;
  advance(phase: string): void;
  settle(result: RaceResult): void;
  mark_won(): void;
  fail(message: string): void;
  reset(): void;
}

function make_lane_view(lane: "cpu" | "gpu"): LaneView {
  const panel = require_el<HTMLElement>(`#panel-${lane}`);
  const backend = require_el<HTMLElement>(`#${lane}-backend`);
  const phase = require_el<HTMLElement>(`#${lane}-phase`);
  const timer = require_el<HTMLElement>(`#${lane}-timer`);
  const caption = require_el<HTMLElement>(`#${lane}-caption`);
  const bar = require_el<HTMLElement>(`#${lane}-bar`);

  let completed = 0;
  // The ticking value is elapsed wall-clock, shown to integer ms so it reads as
  // a fast counter; the settled median is shown to one decimal, so the registers
  // are visually distinct as well as captioned distinctly.
  const stopwatch = make_stopwatch((elapsed_ms) => {
    timer.textContent = elapsed_ms.toFixed(0);
  });

  function set_progress(fraction: number): void {
    bar.style.width = `${(fraction * 100).toFixed(1)}%`;
  }

  return {
    begin(): void {
      completed = 0;
      panel.classList.remove("race-panel--won", "race-panel--failed");
      phase.textContent = "initializing";
      caption.textContent = "elapsed";
      set_progress(0.04);
      stopwatch.start();
    },
    advance(phase_name: string): void {
      completed += 1;
      phase.textContent = phase_name === "warmup" ? "warming up" : "timing";
      // Never reach 100% on progress alone — the bar fills to full only when the
      // measured result lands, so a full bar always means "done," not "almost."
      set_progress(Math.min(completed / EXPECTED_STEPS, 0.98));
    },
    settle(result: RaceResult): void {
      stopwatch.freeze();
      // Honest relabel: a 'webgpu' lane that fell back reports 'webgl' here, so
      // the panel never claims a backend it did not run.
      backend.textContent = result.actual_backend.toUpperCase();
      phase.textContent = "done";
      caption.textContent = `median · ${result.reps_ms.length} reps`;
      timer.textContent = result.median_ms.toFixed(1);
      set_progress(1);
    },
    mark_won(): void {
      panel.classList.add("race-panel--won");
    },
    fail(message: string): void {
      stopwatch.freeze();
      panel.classList.add("race-panel--failed");
      phase.textContent = "failed";
      caption.textContent = message;
      timer.textContent = "—";
    },
    reset(): void {
      stopwatch.reset();
      completed = 0;
      panel.classList.remove("race-panel--won", "race-panel--failed");
      phase.textContent = "ready";
      caption.textContent = "ready";
      timer.textContent = "0";
      set_progress(0);
    },
  };
}

export interface RaceUi {
  run(): Promise<void>;
}

export function make_race_ui(): RaceUi {
  const run_button = require_el<HTMLButtonElement>("#run-race");
  const cpu_view = make_lane_view("cpu");
  const gpu_view = make_lane_view("gpu");
  const cpu_canvas = require_el<HTMLCanvasElement>("#cpu-scatter");
  const gpu_canvas = require_el<HTMLCanvasElement>("#gpu-scatter");

  const tile_cpu_ms = require_el<HTMLElement>("#tile-cpu-ms");
  const tile_gpu_ms = require_el<HTMLElement>("#tile-gpu-ms");
  const tile_speedup = require_el<HTMLElement>("#tile-speedup");
  const tile_speedup_label = require_el<HTMLElement>("#tile-speedup-label");
  const tile_throughput = require_el<HTMLElement>("#tile-throughput");

  const parity = require_el<HTMLElement>("#parity");
  const parity_text = require_el<HTMLElement>("#parity-text");

  const first_run_toggle = require_el<HTMLInputElement>("#first-run-toggle");
  const first_run_readout = require_el<HTMLElement>("#first-run-readout");

  // The cold first-run number lives ONLY behind this toggle and is never read by
  // the headline multiplier, so it cannot leak into the "N.Nx faster" claim.
  first_run_toggle.addEventListener("change", () => {
    first_run_readout.hidden = !first_run_toggle.checked;
  });

  function view_for(lane: BackendLane): LaneView {
    return lane === "cpu" ? cpu_view : gpu_view;
  }

  function render_scatters(dataset: MakeBlobsResult): void {
    const projection = project_2d_pca(
      dataset.data,
      DEFAULT_RACE_CONFIG.n_samples,
      DEFAULT_RACE_CONFIG.n_features,
    );
    render_scatter(cpu_canvas, projection, dataset.labels);
    render_scatter(gpu_canvas, projection, dataset.labels);
  }

  function reset_headline(): void {
    for (const tile of [tile_cpu_ms, tile_gpu_ms, tile_speedup, tile_throughput]) {
      tile.textContent = "—";
    }
    tile_speedup_label.textContent = "speedup";
    parity.dataset.state = "idle";
    parity_text.textContent = "Racing…";
    first_run_toggle.checked = false;
    first_run_readout.hidden = true;
    first_run_readout.textContent = "";
  }

  function render_headline(
    outcome: RaceOutcome,
    cpu_result: RaceResult,
    gpu_result: RaceResult,
  ): void {
    tile_cpu_ms.textContent = format_ms(cpu_result.median_ms);
    tile_gpu_ms.textContent = format_ms(gpu_result.median_ms);

    const gpu_name = outcome.gpu_backend.toUpperCase();
    if (outcome.speedup >= 1) {
      tile_speedup.textContent = `${outcome.speedup.toFixed(1)}×`;
      tile_speedup_label.textContent = `${gpu_name} is faster`;
      gpu_view.mark_won();
    } else {
      // Small-n CPU win is first-class, not a bug — show it as a CPU multiplier
      // rather than a sub-1 number that reads as broken.
      tile_speedup.textContent = `${(1 / outcome.speedup).toFixed(1)}×`;
      tile_speedup_label.textContent = "CPU is faster";
      cpu_view.mark_won();
    }

    const winner = outcome.speedup >= 1 ? gpu_result : cpu_result;
    tile_throughput.textContent = `${format_count(winner.points_per_sec)} pts/s`;

    if (results_agree(cpu_result.result_checksum, gpu_result.result_checksum)) {
      parity.dataset.state = "match";
      // "matching checksum", not "identical matrix": the cross-lane signal is a
      // scalar sum of the affinity entries, which agreeing proves the same
      // kernel ran, not that all 4M entries are bit-identical.
      parity_text.textContent =
        "Same result — both backends agree on the RBF affinity matrix (matching checksum).";
    } else {
      parity.dataset.state = "diverged";
      parity_text.textContent =
        "Results diverged — one backend fell back or computed a different kernel.";
    }

    first_run_readout.textContent =
      `First run incl. shader compile — CPU ${format_ms(cpu_result.first_run_ms)}, ` +
      `${gpu_name} ${format_ms(gpu_result.first_run_ms)}. Excluded from the multiplier above.`;
  }

  function render_failure(error: unknown): void {
    tile_speedup.textContent = "—";
    tile_speedup_label.textContent = "race incomplete";
    // Distinct from "diverged": a crashed or timed-out lane yields no comparison
    // at all, so it must not wear the verdict colour that means "results
    // disagreed".
    parity.dataset.state = "error";
    parity_text.textContent =
      error instanceof Error
        ? `Race incomplete: ${error.message}`
        : "Race incomplete — a lane failed.";
  }

  async function run(): Promise<void> {
    run_button.disabled = true;
    cpu_view.reset();
    gpu_view.reset();
    reset_headline();

    let cpu_result: RaceResult | undefined;
    let gpu_result: RaceResult | undefined;

    try {
      const outcome = await run_race(DEFAULT_RACE_CONFIG, {
        on_dataset: render_scatters,
        on_progress: (lane, phase) => {
          if (phase === "init") view_for(lane).begin();
          else view_for(lane).advance(phase);
        },
        on_lane_result: (result) => {
          if (result.requested_lane === "cpu") cpu_result = result;
          else gpu_result = result;
          view_for(result.requested_lane).settle(result);
        },
        on_lane_error: (lane, message) => view_for(lane).fail(message),
      });
      // Both results are guaranteed present once run_race resolves (it rejects
      // if either lane failed), so the headline always has both medians.
      if (cpu_result && gpu_result) {
        render_headline(outcome, cpu_result, gpu_result);
      }
    } catch (error) {
      render_failure(error);
    } finally {
      run_button.disabled = false;
    }
  }

  return { run };
}
