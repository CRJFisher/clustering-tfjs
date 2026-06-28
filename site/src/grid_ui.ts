import { run_grid } from "./grid";
import type { GridDatasets } from "./grid";
import {
  GRID_ALGORITHMS,
  GRID_CELLS,
  GRID_DATASETS,
  cell_id_of,
  count_parity,
} from "./grid_config";
import type { GridCell, GridParams, ParityTier } from "./grid_config";
import type { GridJob } from "./grid_protocol";
import { params_equal, resolve_all } from "./grid_controls";
import type { ControlOverrides } from "./grid_controls";
import { make_grid_controls } from "./grid_controls_ui";
import type { ToyDataset } from "./make_toy_datasets";
import { render_scatter } from "./scatter_canvas";
import type { Projection2d } from "./project_2d";

// The parity-grid fold. It builds the 5×5 DOM from grid_config (one source of
// truth, so a header can never mislabel a cell), kicks off the single worker, and
// renders each cell's scatter ONCE the instant its labels stream back — the same
// render-once discipline as the race scatter, no animation loop. Every published
// label comes straight from the worker; the main thread never clusters.

const GRID_CELL_RADIUS = 1.6;

// The glyph + short caption + hover explanation each parity tier advertises,
// mirroring the race fold's precise-over-salesy copy.
const PARITY_COPY: Record<
  ParityTier,
  { glyph: string; short: string; title: string }
> = {
  matches: {
    glyph: "✓",
    short: "matches sklearn",
    title:
      "These labels match scikit-learn's for this algorithm and shape — parity, " +
      "including the shapes where the algorithm is meant to fail (K-Means cuts " +
      "straight through moons and circles, exactly as scikit-learn's does).",
  },
  drifts: {
    glyph: "≈",
    short: "cores match · float32 drift",
    title:
      "Cluster cores match scikit-learn; a few boundary points can differ under " +
      "float32 (Spectral's eigen-embedding / HDBSCAN's density tree). We annotate " +
      "the difference rather than tune it away.",
  },
  "no-reference": {
    glyph: "◇",
    short: "library-only (no sklearn ref)",
    title: "SOM has no scikit-learn counterpart in this comparison.",
  },
  "no-truth": {
    glyph: "∅",
    short: "no true clusters",
    title:
      "Uniform data has no real clusters. HDBSCAN correctly reports all-noise; " +
      "the partitioning methods are forced to invent an arbitrary split.",
  },
};

// The hover explanation for an exploratory (off-Auto) cell, which replaces the
// parity tier's title so the tooltip can never assert a scikit-learn parity the
// overridden params were never checked for.
const OVERRIDDEN_TITLE =
  "Your parameters — this exact combination was never checked against " +
  "scikit-learn, so no parity is claimed.";

function require_el<T extends HTMLElement>(selector: string): T {
  const element = document.querySelector<T>(selector);
  if (!element) throw new Error(`Missing required element ${selector}`);
  return element;
}

// A cell renders the dataset's two standardized columns directly — the points are
// already 2-D, so unlike the race scatter there is no PCA to run.
function projection_of(dataset: ToyDataset): Projection2d {
  const x = new Float32Array(dataset.n_samples);
  const y = new Float32Array(dataset.n_samples);
  for (let i = 0; i < dataset.n_samples; i++) {
    x[i] = dataset.data[i * 2];
    y[i] = dataset.data[i * 2 + 1];
  }
  return { x, y };
}

interface CellView {
  figure: HTMLElement;
  canvas: HTMLCanvasElement;
  annotation: HTMLElement;
  cell: GridCell;
  // "Two moons clustered by Spectral" — the canvas aria-label's stable prefix,
  // extended on each result with the live state so screen readers hear the
  // parity/override outcome, not a label frozen at build time.
  aria_base: string;
}

// Build the grid DOM from config: a corner, five algorithm column headers, then
// per dataset row a row label plus its five cells. Returns the cell views keyed
// by cell_id so streamed results find their canvas in O(1).
function build_grid(container: HTMLElement): Map<string, CellView> {
  const views = new Map<string, CellView>();

  const corner = document.createElement("div");
  corner.className = "demo-grid__corner";
  container.append(corner);

  for (const algorithm of GRID_ALGORITHMS) {
    const head = document.createElement("div");
    head.className = "demo-grid__col-head";
    head.textContent = algorithm.label;
    container.append(head);
  }

  const cell_by_id = new Map(GRID_CELLS.map((cell) => [cell.cell_id, cell]));

  for (const dataset of GRID_DATASETS) {
    const row_head = document.createElement("div");
    row_head.className = "demo-grid__row-head";
    row_head.textContent = dataset.label;
    container.append(row_head);

    for (const algorithm of GRID_ALGORITHMS) {
      const cell_id = cell_id_of(dataset.id, algorithm.id);
      const cell = cell_by_id.get(cell_id);
      if (!cell) throw new Error(`Missing grid cell config for ${cell_id}`);

      const aria_base = `${dataset.label} clustered by ${algorithm.label}`;

      // Each cell is a button: clicking or keyboard-activating it selects the cell
      // and drives the code panel to its runnable source. aria-pressed carries the
      // single-selection state, and the figure is the SOLE accessible-name carrier
      // (live-updated with the result), so the inner canvas stays presentational —
      // no nested img role or duplicate label to confuse a screen reader. tabIndex
      // starts at -1; one cell is made tabbable (roving tabindex) so the grid is a
      // single tab stop, with arrow keys moving focus within it.
      const figure = document.createElement("figure");
      figure.className = "demo-grid__cell";
      figure.dataset.state = "pending";
      figure.dataset.parity = cell.parity;
      figure.dataset.cellId = cell_id;
      figure.setAttribute("role", "button");
      figure.tabIndex = -1;
      figure.setAttribute("aria-pressed", "false");
      figure.setAttribute("aria-label", aria_base);

      const canvas = document.createElement("canvas");
      canvas.className = "demo-grid__scatter";
      canvas.width = 150;
      canvas.height = 150;

      const annotation = document.createElement("figcaption");
      annotation.className = "demo-grid__annotation";
      annotation.title = PARITY_COPY[cell.parity].title;
      annotation.textContent = "…";

      figure.append(canvas, annotation);
      container.append(figure);
      views.set(cell_id, { figure, canvas, annotation, cell, aria_base });
    }
  }

  return views;
}

function found_text(
  cell: GridCell,
  n_clusters_found: number,
  noise_count: number,
): string {
  if (cell.algorithm_id === "hdbscan" && cell.parity === "no-truth") {
    return noise_count > 0 ? `${noise_count} noise` : `k=${n_clusters_found}`;
  }
  return `k=${n_clusters_found}${noise_count > 0 ? ` · ${noise_count} noise` : ""}`;
}

// In Auto a cell wears its scikit-learn parity glyph + claim. Off Auto its params
// were never checked against scikit-learn, so the claim is replaced by a neutral
// "your params" — the page never asserts a parity it did not verify.
function annotation_text(
  cell: GridCell,
  n_clusters_found: number,
  noise_count: number,
  overridden: boolean,
): string {
  const found = found_text(cell, n_clusters_found, noise_count);
  if (overridden) return `✎ ${found} · not checked vs sklearn`;
  return `${PARITY_COPY[cell.parity].glyph} ${found} · ${PARITY_COPY[cell.parity].short}`;
}

// A drag streams control changes; each re-cluster is fast but re-fitting on every
// pixel of travel would still flood the worker, so changes settle for this long
// before the resting params are fit. Combined with single-flight below, the worker
// never has more than one batch in flight.
const RECLUSTER_DEBOUNCE_MS = 140;

// Only a backend that actually came up may drive live controls. The init failure
// labels (`grid.ts`) are the sentinels to exclude.
function backend_is_live(actual_backend: string): boolean {
  return (
    actual_backend !== "none" &&
    actual_backend !== "timed out" &&
    !actual_backend.startsWith("worker error")
  );
}

// Arrow-key steps through the row-major grid: ±1 within a row, ±column-count
// between rows. Selection follows EXPLICIT activation (Enter/Space), not focus, so
// arrowing across cells doesn't fire a re-render per cell.
const GRID_COLUMNS = GRID_ALGORITHMS.length;
const ARROW_STEPS: Record<string, number> = {
  ArrowLeft: -1,
  ArrowRight: 1,
  ArrowUp: -GRID_COLUMNS,
  ArrowDown: GRID_COLUMNS,
};

// What main.ts drives to mirror the grid in the code panel and the permalink: read
// the live overrides + selected cell, subscribe to their changes and to the live
// backend, and restore a shared selection/overrides. The on_* sinks hold a SINGLE
// callback (main is the only consumer); a second consumer would need an emitter.
export interface GridSection {
  get_overrides: () => ControlOverrides;
  get_selected_cell: () => string | null;
  on_selection_change: (cb: (cell_id: string) => void) => void;
  on_overrides_change: (cb: (overrides: ControlOverrides) => void) => void;
  on_backend: (cb: (actual_backend: string, live: boolean) => void) => void;
  select_cell: (cell_id: string) => void;
  apply_overrides: (overrides: ControlOverrides) => void;
}

export function make_grid_ui(): GridSection {
  const container = require_el<HTMLElement>("#demo-grid");
  const status = require_el<HTMLElement>("#grid-status");
  const backend = require_el<HTMLElement>("#grid-backend");
  const footnote = require_el<HTMLElement>("#grid-footnote");
  const controls_mount = require_el<HTMLElement>("#grid-controls");

  const views = build_grid(container);
  let datasets: GridDatasets | undefined;

  // Single-callback sinks (main.ts is the only consumer); genuinely absent until
  // registered, so null is the honest empty state.
  let selection_listener: ((cell_id: string) => void) | null = null;
  let overrides_listener: ((overrides: ControlOverrides) => void) | null = null;
  let backend_listener:
    | ((actual_backend: string, live: boolean) => void)
    | null = null;
  let selected_cell_id: string | null = null;
  const ordered_cell_ids = GRID_CELLS.map((cell) => cell.cell_id);

  // Roving tabindex: exactly one cell is in the tab order at a time, so the whole
  // grid is a single Tab stop. The first cell starts tabbable; selecting or
  // arrowing moves the tabbable one so Tab always lands back on the last cell used.
  let tabbable_cell_id = ordered_cell_ids[0];
  views.get(tabbable_cell_id)?.figure.setAttribute("tabindex", "0");

  function set_tabbable(cell_id: string): void {
    if (cell_id === tabbable_cell_id) return;
    const next = views.get(cell_id);
    if (!next) return;
    views.get(tabbable_cell_id)?.figure.setAttribute("tabindex", "-1");
    next.figure.setAttribute("tabindex", "0");
    tabbable_cell_id = cell_id;
  }

  // Move the selection to a cell, updating aria-pressed on the outgoing and
  // incoming figures and announcing the change. Focus is left untouched — a
  // programmatic restore must not yank focus or scroll the grid into view — but the
  // selected cell becomes the tab stop so a later Tab lands on it.
  function set_selected(cell_id: string): void {
    if (cell_id === selected_cell_id) return;
    const next = views.get(cell_id);
    if (!next) return;
    if (selected_cell_id) {
      views.get(selected_cell_id)?.figure.setAttribute("aria-pressed", "false");
    }
    next.figure.setAttribute("aria-pressed", "true");
    selected_cell_id = cell_id;
    set_tabbable(cell_id);
    selection_listener?.(cell_id);
  }

  // One delegated click handler instead of 25: resolve the activated cell from the
  // event target's nearest figure.
  container.addEventListener("click", (event) => {
    const target = event.target;
    if (!(target instanceof HTMLElement)) return;
    const figure = target.closest<HTMLElement>(".demo-grid__cell");
    const cell_id = figure?.dataset.cellId;
    if (cell_id) set_selected(cell_id);
  });

  container.addEventListener("keydown", (event) => {
    const target = event.target;
    if (!(target instanceof HTMLElement)) return;
    const figure = target.closest<HTMLElement>(".demo-grid__cell");
    const cell_id = figure?.dataset.cellId;
    if (!cell_id) return;
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      set_selected(cell_id);
      return;
    }
    const step = ARROW_STEPS[event.key];
    if (step === undefined) return;
    event.preventDefault();
    const index = ordered_cell_ids.indexOf(cell_id);
    const next_index = Math.min(
      ordered_cell_ids.length - 1,
      Math.max(0, index + step),
    );
    const next_id = ordered_cell_ids[next_index];
    // Roving: the cell we move to becomes the tab stop, then takes focus.
    set_tabbable(next_id);
    views.get(next_id)?.figure.focus();
  });

  // The params each cell was last fit with — seeded from curated (Auto). Diffing
  // the resolved params against this is what tells us which cells a control change
  // actually moved, so a re-cluster only re-fits cells whose params really changed.
  const last_params = new Map<string, GridParams>(
    GRID_CELLS.map((cell) => [cell.cell_id, cell.params]),
  );

  // The footnote count is computed from config, never hand-typed, so it can never
  // claim more matching cells than the grid actually advertises.
  const matches = count_parity("matches");
  const drifts = count_parity("drifts");
  footnote.textContent =
    `${matches} of ${GRID_CELLS.length} cells match scikit-learn's labels exactly — ` +
    `including where K-Means is meant to fail (the same straight cut scikit-learn makes ` +
    `through moons and circles): matching scikit-learn is the claim, not getting every ` +
    `shape "right". The ${drifts} cells marked ≈ — Spectral on the curved and anisotropic ` +
    `rows, plus HDBSCAN — match the cluster cores but can disagree on a handful of boundary ` +
    `points under float32, annotated rather than tuned away. SOM has no scikit-learn ` +
    `counterpart. Fixed seeds make every visitor's grid identical.`;

  // The parity decision each cell's IN-FLIGHT result was dispatched under. The
  // rendered labels come from the job's params (snapshotted at dispatch), so the
  // badge must too — recomputing it against live overrides could paint a ✓ over
  // labels fit with different params if a control moved between request and result.
  const dispatched_overridden = new Map<string, boolean>(
    GRID_CELLS.map((cell) => [cell.cell_id, false]),
  );

  // The CSS state AND the hover tooltip both follow the override flag, so an
  // exploratory cell can never keep a parity tooltip that asserts a scikit-learn
  // match its params were never checked for.
  function set_cell_explore(view: CellView, overridden: boolean): void {
    view.figure.dataset.explore = overridden ? "true" : "false";
    view.annotation.title = overridden
      ? OVERRIDDEN_TITLE
      : PARITY_COPY[view.cell.parity].title;
  }

  const panel = make_grid_controls(controls_mount, () => {
    schedule_recluster();
    // Notify on every settled control change so the code panel re-renders the
    // selected cell against the live params (even when the selected cell itself
    // didn't move, its resolved params may have).
    overrides_listener?.(panel.get_overrides());
  });

  const controller = run_grid({
    on_datasets: (generated) => {
      datasets = generated;
    },
    on_progress: (completed, total) => {
      status.textContent =
        completed < total
          ? `Computing ${completed} / ${total} cells…`
          : `${total} / ${total} cells computed`;
    },
    on_cell_result: (cell_id, labels, n_clusters_found, noise_count) => {
      const view = views.get(cell_id);
      if (!view || !datasets) return;
      const dataset = datasets.by_id.get(view.cell.dataset_id);
      if (!dataset) return;
      // The state these labels were computed under, not whatever the controls read
      // right now (which may have moved on while the fit was in flight).
      const overridden = dispatched_overridden.get(cell_id) ?? false;
      render_scatter(view.canvas, projection_of(dataset), labels, {
        point_radius: GRID_CELL_RADIUS,
      });
      set_cell_explore(view, overridden);
      view.annotation.textContent = annotation_text(
        view.cell,
        n_clusters_found,
        noise_count,
        overridden,
      );
      // Re-state the live outcome on the cell button so a screen reader hears the
      // parity/override result, not the label frozen at build time.
      view.figure.setAttribute(
        "aria-label",
        overridden
          ? `${view.aria_base} — your params, k=${n_clusters_found}`
          : `${view.aria_base} — ${PARITY_COPY[view.cell.parity].short}, k=${n_clusters_found}`,
      );
      view.figure.dataset.state = "done";
    },
    on_cell_error: (cell_id, message) => {
      const view = views.get(cell_id);
      if (!view) return;
      view.annotation.textContent = `failed: ${message}`;
      view.figure.dataset.state = "error";
    },
    on_done: (actual_backend) => {
      backend.textContent = actual_backend.toUpperCase();
      // Surface the raw backend label (and whether it's live) so the code panel can
      // show the real init arg and main can run a deferred permalink restore once
      // the controls are warm.
      backend_listener?.(actual_backend, backend_is_live(actual_backend));
      // Terminal sweep. A mid-stream timeout or worker crash terminates the worker
      // with cells still unreported, so any cell left "pending" would sit dimmed
      // and "computing" forever and the status line would freeze. Mark every
      // still-pending cell failed and make the status reflect what actually
      // finished — never a frozen "Computing N / 25" over a half-dimmed grid.
      let done = 0;
      let failed = 0;
      for (const view of views.values()) {
        if (view.figure.dataset.state === "done") {
          done += 1;
          continue;
        }
        failed += 1;
        if (view.figure.dataset.state === "pending") {
          view.figure.dataset.state = "error";
          view.annotation.textContent = "didn't finish";
        }
      }
      status.textContent =
        failed === 0
          ? `${done} / ${GRID_CELLS.length} cells computed`
          : `${done} / ${GRID_CELLS.length} cells computed · ${failed} failed`;
      // The live controls only make sense while a backend is warm and holding the
      // datasets. A live backend enables them; a failed init OR a later worker
      // crash (on_done fires again with a non-live label) disables them and clears
      // any stuck in-flight latch, so the controls never sit enabled over a dead
      // worker silently swallowing every re-cluster.
      if (backend_is_live(actual_backend)) {
        panel.set_enabled(true);
      } else {
        panel.set_enabled(false);
        recluster_in_flight = false;
      }
    },
    on_recluster_done: () => {
      recluster_in_flight = false;
      // A newer control position arrived mid-fit; fit the latest now.
      if (recluster_dirty) {
        recluster_dirty = false;
        flush_recluster();
      }
    },
  });

  // Single-flight re-cluster scheduler. A control change updates each cell's
  // parity-drop badge immediately, diffs resolved-vs-last params to find the moved
  // cells, and re-fits only those — never more than one batch in flight, the rest
  // coalesced into the latest resting position.
  let recluster_in_flight = false;
  let recluster_dirty = false;
  let debounce_handle: ReturnType<typeof setTimeout> | undefined;

  function flush_recluster(): void {
    if (recluster_in_flight) {
      recluster_dirty = true;
      return;
    }
    const overrides = panel.get_overrides();
    const jobs: GridJob[] = [];
    for (const resolved of resolve_all(overrides)) {
      const view = views.get(resolved.cell.cell_id);
      // Drop or restore the parity badge the instant the control moves, before the
      // fit even starts, so the grid never shows a stale ✓ over your params.
      if (view) set_cell_explore(view, resolved.overridden);
      const prev = last_params.get(resolved.cell.cell_id);
      if (prev && params_equal(prev, resolved.params)) continue;
      last_params.set(resolved.cell.cell_id, resolved.params);
      // Snapshot the parity decision for the result this job will produce.
      dispatched_overridden.set(resolved.cell.cell_id, resolved.overridden);
      if (view) {
        view.figure.dataset.state = "pending";
        view.annotation.textContent = "…";
      }
      jobs.push({
        cell_id: resolved.cell.cell_id,
        dataset_id: resolved.cell.dataset_id,
        algorithm_id: resolved.cell.algorithm_id,
        params: resolved.params,
      });
    }
    if (jobs.length === 0) return;
    recluster_in_flight = true;
    controller.recluster(jobs);
  }

  function schedule_recluster(): void {
    if (debounce_handle !== undefined) clearTimeout(debounce_handle);
    debounce_handle = setTimeout(() => {
      debounce_handle = undefined;
      flush_recluster();
    }, RECLUSTER_DEBOUNCE_MS);
  }

  return {
    get_overrides: () => panel.get_overrides(),
    get_selected_cell: () => selected_cell_id,
    on_selection_change: (cb) => {
      selection_listener = cb;
    },
    on_overrides_change: (cb) => {
      overrides_listener = cb;
    },
    on_backend: (cb) => {
      backend_listener = cb;
    },
    select_cell: (cell_id) => set_selected(cell_id),
    apply_overrides: (overrides) => panel.apply_overrides(overrides),
  };
}
