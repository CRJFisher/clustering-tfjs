import { create_grid_runner } from "./grid";
import type { GridDatasets, GridRunner } from "./grid";
import {
  GRID_ALGORITHMS,
  GRID_CELLS,
  GRID_DATASETS,
  cell_id_of,
} from "./grid_config";
import type { GridCell } from "./grid_config";
import type { ToyDataset } from "./make_toy_datasets";
import { render_scatter } from "./scatter_canvas";
import type { Projection2d } from "./project_2d";

// The clustering-grid fold. It builds the 5×5 DOM from grid_config (one source of
// truth, so a header can never mislabel a cell), and on `populate()` kicks off the
// single worker and renders each cell's scatter ONCE the instant its labels stream
// back — no animation loop. Every published label comes straight from the worker;
// the main thread never clusters.

const GRID_CELL_RADIUS = 1.6;

function require_el<T extends HTMLElement>(selector: string): T {
  const element = document.querySelector<T>(selector);
  if (!element) throw new Error(`Missing required element ${selector}`);
  return element;
}

// A cell renders the dataset's two standardized columns directly — the points are
// already 2-D, so there is no projection to run.
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
  // extended on each result with the live count.
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
      // (live-updated with the result), so the inner canvas stays presentational.
      // tabIndex starts at -1; one cell is made tabbable (roving tabindex) so the
      // grid is a single tab stop, with arrow keys moving focus within it.
      const figure = document.createElement("figure");
      figure.className = "demo-grid__cell";
      figure.dataset.state = "idle";
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
      annotation.textContent = "—";

      figure.append(canvas, annotation);
      container.append(figure);
      views.set(cell_id, { figure, canvas, annotation, cell, aria_base });
    }
  }

  return views;
}

// The per-cell result line: how many clusters the algorithm found, plus the noise
// count when HDBSCAN labelled points as belonging to none.
function found_text(n_clusters_found: number, noise_count: number): string {
  return `k=${n_clusters_found}${noise_count > 0 ? ` · ${noise_count} noise` : ""}`;
}

// Arrow-key steps through the row-major grid: ±1 within a row, ±column-count
// between rows. Selection follows EXPLICIT activation (Enter/Space), not focus.
const GRID_COLUMNS = GRID_ALGORITHMS.length;
const ARROW_STEPS: Record<string, number> = {
  ArrowLeft: -1,
  ArrowRight: 1,
  ArrowUp: -GRID_COLUMNS,
  ArrowDown: GRID_COLUMNS,
};

// What main.ts drives to mirror the grid in the code panel and the permalink: read
// the selected cell, subscribe to its changes and to the live backend, restore a
// shared selection, and kick off the clustering sweep on demand.
export interface GridSection {
  get_selected_cell: () => string | null;
  on_selection_change: (cb: (cell_id: string) => void) => void;
  on_backend: (cb: (actual_backend: string, live: boolean) => void) => void;
  select_cell: (cell_id: string) => void;
  populate: () => void;
}

export function make_grid_ui(): GridSection {
  const container = require_el<HTMLElement>("#demo-grid");
  const status = require_el<HTMLElement>("#grid-status");
  const backend = require_el<HTMLElement>("#grid-backend");
  const footnote = require_el<HTMLElement>("#grid-footnote");

  const views = build_grid(container);
  let datasets: GridDatasets | undefined;

  // Single-callback sinks (main.ts is the only consumer); genuinely absent until
  // registered, so null is the honest empty state.
  let selection_listener: ((cell_id: string) => void) | null = null;
  let backend_listener:
    | ((actual_backend: string, live: boolean) => void)
    | null = null;
  let selected_cell_id: string | null = null;

  // Spawn the worker and warm the backend NOW, at page load — long before the
  // visitor clicks to cluster. The backend label fills in as soon as init
  // resolves, and the fits later run on the already-warm engine.
  backend.textContent = "initializing…";
  const runner: GridRunner = create_grid_runner({
    on_backend_ready: (actual_backend) => {
      backend.textContent = actual_backend.toUpperCase();
      // A resolved backend is always a runnable one; mirror it into the code panel.
      backend_listener?.(actual_backend, true);
    },
    on_backend_error: (message) => {
      backend.textContent = "unavailable";
      footnote.textContent = `Backend init failed: ${message}`;
    },
  });
  const ordered_cell_ids = GRID_CELLS.map((cell) => cell.cell_id);

  // Roving tabindex: exactly one cell is in the tab order at a time, so the whole
  // grid is a single Tab stop.
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
  // incoming figures. Focus is left untouched — a programmatic restore must not
  // yank focus — but the selected cell becomes the tab stop.
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
    set_tabbable(next_id);
    views.get(next_id)?.figure.focus();
  });

  // Fixed seeds make every visitor's grid identical. The note describes the
  // behaviour on show, not a comparison against any reference library.
  footnote.textContent =
    `${GRID_CELLS.length} cells: every dataset (rows) clustered by every algorithm ` +
    `(columns) on fixed seeds, so the grid is identical for every visitor. K-Means ` +
    `partitions by straight cuts; Spectral and single-linkage Agglomerative follow ` +
    `curved manifolds; HDBSCAN discovers the cluster count from density and marks ` +
    `sparse points as noise; SOM trains a neuron lattice then merges it down to k.`;

  // Idempotent: the grid clusters once per page. A second populate() (e.g. a
  // double-click on the button) is ignored so it never spawns a second worker.
  let started = false;

  function populate(): void {
    if (started) return;
    started = true;
    status.textContent = `Computing 0 / ${GRID_CELLS.length} cells…`;
    for (const view of views.values()) {
      view.figure.dataset.state = "pending";
      view.annotation.textContent = "…";
    }

    runner.run({
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
        render_scatter(view.canvas, projection_of(dataset), labels, {
          point_radius: GRID_CELL_RADIUS,
        });
        view.annotation.textContent = found_text(n_clusters_found, noise_count);
        view.figure.setAttribute(
          "aria-label",
          `${view.aria_base} — k=${n_clusters_found}`,
        );
        view.figure.dataset.state = "done";
      },
      on_cell_error: (cell_id, message) => {
        const view = views.get(cell_id);
        if (!view) return;
        view.annotation.textContent = `failed: ${message}`;
        view.figure.dataset.state = "error";
      },
      on_done: () => {
        // The backend label was already filled in at init (on_backend_ready), so a
        // terminal sweep only reconciles the cells. A mid-stream timeout or crash
        // leaves cells unreported: mark every still-pending cell failed and make the
        // status reflect what actually finished, never a frozen "Computing N / 25".
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
      },
    });
  }

  return {
    get_selected_cell: () => selected_cell_id,
    on_selection_change: (cb) => {
      selection_listener = cb;
    },
    on_backend: (cb) => {
      backend_listener = cb;
    },
    select_cell: (cell_id) => set_selected(cell_id),
    populate,
  };
}
