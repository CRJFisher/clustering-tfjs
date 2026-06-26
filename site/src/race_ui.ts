import { run_race, DEFAULT_RACE_CONFIG } from "./race";
import type { RaceConfig, RaceOutcome } from "./race";
import type { BackendLane, RaceResult } from "./race_protocol";
import type { MakeBlobsResult } from "./make_blobs_js";
import { project_2d_pca } from "./project_2d";
import { render_scatter } from "./scatter_canvas";
import { make_stopwatch } from "./stopwatch";
import {
  make_crossover_estimator,
  N_MAX,
  N_MIN,
} from "./crossover";

// The whole race fold. The controller is a pure orchestrator over the existing
// `run_race` harness: it translates the harness's callbacks into cheap DOM
// updates (timers, bars, tiles) and renders the scatter ONCE, before timing.
// Every published number is read straight from the worker's `RaceResult`; the
// main thread never times compute, so it can never present its own wall-clock as
// a measured figure.

// Both lanes emit `warmups + reps` progress events after init; the racing bar
// advances one step per event toward this total. The schedule is fixed across n,
// so this stays constant regardless of the slider position.
const EXPECTED_STEPS = DEFAULT_RACE_CONFIG.warmups + DEFAULT_RACE_CONFIG.reps;

// Dragging the slider fires a stream of `input` events; a race spawns two workers
// and runs warmups+reps, so re-racing on every event would be wasteful and janky.
// We let the drag settle for this long, then race the n it rests at.
const RACE_DEBOUNCE_MS = 280;

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

// Main-thread feature probe for the pre-race banner. The worker runs its own
// authoritative check before initializing; this only decides whether to warn and
// reveal the reference clip BEFORE the first race, so the visitor never stares at
// a "WebGPU" panel the browser cannot honour. The post-race `actual_backend`
// reconciliation is the source of truth.
function has_webgpu(): boolean {
  const nav: Navigator & { gpu?: unknown } = navigator;
  return nav.gpu != null;
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

export function make_race_ui(): void {
  const run_button = require_el<HTMLButtonElement>("#run-race");
  const slider = require_el<HTMLInputElement>("#n-slider");
  const slider_value = require_el<HTMLElement>("#n-value");
  const crossover_mark = require_el<HTMLElement>("#crossover-mark");
  const crossover_caption = require_el<HTMLElement>("#crossover-caption");
  const cpu_view = make_lane_view("cpu");
  const gpu_view = make_lane_view("gpu");
  const cpu_canvas = require_el<HTMLCanvasElement>("#cpu-scatter");
  const gpu_canvas = require_el<HTMLCanvasElement>("#gpu-scatter");

  const crossover = make_crossover_estimator();

  const tile_cpu_ms = require_el<HTMLElement>("#tile-cpu-ms");
  const tile_gpu_ms = require_el<HTMLElement>("#tile-gpu-ms");
  const tile_speedup = require_el<HTMLElement>("#tile-speedup");
  const tile_speedup_label = require_el<HTMLElement>("#tile-speedup-label");
  const tile_throughput = require_el<HTMLElement>("#tile-throughput");

  const parity = require_el<HTMLElement>("#parity");
  const parity_text = require_el<HTMLElement>("#parity-text");

  const first_run_toggle = require_el<HTMLInputElement>("#first-run-toggle");
  const first_run_readout = require_el<HTMLElement>("#first-run-readout");

  const gpu_backend_label = require_el<HTMLElement>("#gpu-backend");
  const fallback_banner = require_el<HTMLElement>("#fallback-banner");
  const fallback_banner_text = require_el<HTMLElement>("#fallback-banner-text");
  const reference_figure = require_el<HTMLElement>("#race-reference");
  const reference_media = require_el<HTMLImageElement>("#race-reference-media");

  const config_n = require_el<HTMLElement>("#config-n");
  const config_cpu_backend = require_el<HTMLElement>("#config-cpu-backend");
  const config_gpu_backend = require_el<HTMLElement>("#config-gpu-backend");
  const config_tfjs_version = require_el<HTMLElement>("#config-tfjs-version");

  // Seed the methodology panel's fixed fields from the one config the workers
  // actually run, so the documented schedule can never drift from the code.
  require_el<HTMLElement>("#methodology-warmups").textContent = String(
    DEFAULT_RACE_CONFIG.warmups,
  );
  require_el<HTMLElement>("#methodology-reps").textContent = String(
    DEFAULT_RACE_CONFIG.reps,
  );
  require_el<HTMLElement>("#config-d").textContent = String(
    DEFAULT_RACE_CONFIG.n_features,
  );
  require_el<HTMLElement>("#config-schedule").textContent =
    `${DEFAULT_RACE_CONFIG.warmups} · ${DEFAULT_RACE_CONFIG.reps}`;

  // The reference clip is base-aware (Pages serves under a sub-path) and only
  // attached when first revealed, so its animation does not run off-screen.
  function reveal_reference(): void {
    if (!reference_figure.hidden) return;
    reference_media.src = `${import.meta.env.BASE_URL}race-reference.svg`;
    reference_figure.hidden = false;
  }

  // Authoritative reconciliation against what the GPU worker actually ran. A
  // 'webgpu' result clears the notice; anything else means the lane fell back,
  // so the banner names the real backend and the reference clip is revealed.
  function reconcile_fallback(gpu_backend: string): void {
    if (gpu_backend === "webgpu") {
      fallback_banner.hidden = true;
      return;
    }
    const name = gpu_backend.toUpperCase();
    fallback_banner_text.textContent =
      `WebGPU isn't available in this browser — the GPU lane is running ${name}, ` +
      "your fastest available backend. The timings below are real, measured on your hardware.";
    fallback_banner.hidden = false;
    reveal_reference();
  }

  // Before any race, navigator.gpu is the best signal: when it is absent the GPU
  // panel must not advertise "WebGPU", and the visitor should already see both
  // the honest notice and the reference payoff.
  if (!has_webgpu()) {
    gpu_backend_label.textContent = "GPU";
    fallback_banner_text.textContent =
      "WebGPU isn't available in this browser — the GPU lane will race your fastest " +
      "available backend (WebGL → WASM → CPU). The timings below are real, measured on your hardware.";
    fallback_banner.hidden = false;
    reveal_reference();
  }

  function update_config(n_samples: number): void {
    config_n.textContent = format_count(n_samples);
  }

  // The cold first-run number lives ONLY behind this toggle and is never read by
  // the headline multiplier, so it cannot leak into the "N.Nx faster" claim.
  first_run_toggle.addEventListener("change", () => {
    first_run_readout.hidden = !first_run_toggle.checked;
  });

  function view_for(lane: BackendLane): LaneView {
    return lane === "cpu" ? cpu_view : gpu_view;
  }

  function render_scatters(dataset: MakeBlobsResult, n_samples: number): void {
    const projection = project_2d_pca(
      dataset.data,
      n_samples,
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
    // A failed race produced no winner, so the verdict caption must not keep
    // asserting the previous race's outcome. The mark is left untouched — it is a
    // cumulative estimate over every completed race, not a per-race verdict.
    crossover_caption.textContent = "Race incomplete — drag to retry.";
  }

  // The caption flips on the MEASURED winner of this very race (the same
  // outcome.speedup the headline tile reads), never on the estimated crossover.
  // Driving both from one measured number is what guarantees the caption can
  // never contradict the bars the visitor is watching.
  function flip_caption(outcome: RaceOutcome): void {
    crossover_caption.textContent =
      outcome.speedup >= 1
        ? "GPU pulls ahead — the O(n²·d) affinity compute now dwarfs the GPU's fixed overhead."
        : "At small n, CPU wins — GPU transfer + dispatch cost more than the math.";
  }

  // Renders ONLY what the visitor has empirically bracketed: a tick lands at the
  // interpolated crossover only once they have raced both a CPU-winning and a
  // GPU-winning n. Otherwise a directional hint points toward the unraced side —
  // the page never marks a tick at an n the visitor has not actually measured.
  function update_crossover_mark(): void {
    const state = crossover.estimate();
    crossover_mark.dataset.state = state.kind;
    if (state.kind === "bracketed") {
      const fraction = (state.n - N_MIN) / (N_MAX - N_MIN);
      crossover_mark.style.setProperty(
        "--crossover-left",
        `${(fraction * 100).toFixed(1)}%`,
      );
      // Near a track end the centred label would overflow the slider; anchor it
      // inward so the value stays on-screen on narrow viewports.
      crossover_mark.dataset.edge =
        fraction < 0.12 ? "min" : fraction > 0.88 ? "max" : "mid";
      crossover_mark.dataset.label = `crossover ≈ n ${format_count(state.n)}`;
    } else if (state.kind === "below_range") {
      crossover_mark.dataset.label = "← smaller n: CPU wins";
    } else if (state.kind === "above_range") {
      crossover_mark.dataset.label = "larger n: GPU wins →";
    } else {
      crossover_mark.dataset.label = "";
    }
  }

  async function run(n_samples: number): Promise<void> {
    run_button.disabled = true;
    cpu_view.reset();
    gpu_view.reset();
    reset_headline();
    // Pull the verdict caption into the busy state alongside the tiles and parity
    // line, so a re-run never shows the previous race's verdict next to lanes that
    // have already reset to zero.
    crossover_caption.textContent = `Racing n = ${format_count(n_samples)}…`;
    update_config(n_samples);

    const config: RaceConfig = { ...DEFAULT_RACE_CONFIG, n_samples };

    let cpu_result: RaceResult | undefined;
    let gpu_result: RaceResult | undefined;

    try {
      const outcome = await run_race(config, {
        on_dataset: (dataset) => render_scatters(dataset, n_samples),
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
        // The methodology panel reports what THIS race ran: the real backends
        // (after any fallback) and the engine version straight off the result.
        config_cpu_backend.textContent = cpu_result.actual_backend.toUpperCase();
        config_gpu_backend.textContent = gpu_result.actual_backend.toUpperCase();
        config_tfjs_version.textContent = gpu_result.tfjs_version;
        reconcile_fallback(gpu_result.actual_backend);
        crossover.add_sample({
          n: n_samples,
          cpu_ms: cpu_result.median_ms,
          gpu_ms: gpu_result.median_ms,
        });
        update_crossover_mark();
        flip_caption(outcome);
      }
    } catch (error) {
      render_failure(error);
    } finally {
      run_button.disabled = false;
    }
  }

  // Single-flight scheduler. Dragging the slider can request many races; only one
  // ever runs at a time and intermediate targets are coalesced into `pending_n`
  // (latest wins). All compute is in workers, so the main thread only ever does
  // the cheap debounce + readout work — the page never freezes, even at n=5000.
  let in_flight = false;
  let pending_n: number | null = null;
  let debounce_handle: ReturnType<typeof setTimeout> | undefined;

  function read_slider_n(): number {
    // The input's min/max/step already bound the value to [200, 5000]; clamp again
    // so a manually-tampered value can never push n past the dense-affinity memory
    // ceiling the cap exists to protect (AC#4).
    return Math.min(N_MAX, Math.max(N_MIN, Number(slider.value)));
  }

  function drain(): void {
    if (in_flight || pending_n === null) return;
    const n = pending_n;
    pending_n = null;
    in_flight = true;
    void run(n).finally(() => {
      in_flight = false;
      // A newer target arrived mid-race: run the latest now.
      if (pending_n !== null) drain();
    });
  }

  function schedule_race(n: number): void {
    pending_n = n;
    if (debounce_handle !== undefined) clearTimeout(debounce_handle);
    debounce_handle = setTimeout(() => {
      debounce_handle = undefined;
      drain();
    }, RACE_DEBOUNCE_MS);
  }

  slider.addEventListener("input", () => {
    const n = read_slider_n();
    // The readout updates synchronously on every event so the number tracks the
    // thumb instantly; the race itself is debounced behind it.
    const formatted = format_count(n);
    slider_value.textContent = formatted;
    update_config(n);
    // Screen readers otherwise announce a bare number; name the unit so the value
    // is meaningful when dragging without sight of the readout.
    slider.setAttribute("aria-valuetext", `${formatted} samples`);
    schedule_race(n);
  });

  run_button.addEventListener("click", () => schedule_race(read_slider_n()));
}
