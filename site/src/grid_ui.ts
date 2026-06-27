import { run_grid } from "./grid";
import type { GridDatasets } from "./grid";
import {
  GRID_ALGORITHMS,
  GRID_CELLS,
  GRID_DATASETS,
  cell_id_of,
  count_parity,
} from "./grid_config";
import type { GridCell, ParityTier } from "./grid_config";
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

      const figure = document.createElement("figure");
      figure.className = "demo-grid__cell";
      figure.dataset.state = "pending";
      figure.dataset.parity = cell.parity;

      const canvas = document.createElement("canvas");
      canvas.className = "demo-grid__scatter";
      canvas.width = 150;
      canvas.height = 150;
      canvas.setAttribute("role", "img");
      canvas.setAttribute(
        "aria-label",
        `${dataset.label} clustered by ${algorithm.label}`,
      );

      const annotation = document.createElement("figcaption");
      annotation.className = "demo-grid__annotation";
      annotation.title = PARITY_COPY[cell.parity].title;
      annotation.textContent = "…";

      figure.append(canvas, annotation);
      container.append(figure);
      views.set(cell_id, { figure, canvas, annotation, cell });
    }
  }

  return views;
}

function annotation_text(
  cell: GridCell,
  n_clusters_found: number,
  noise_count: number,
): string {
  const parity = PARITY_COPY[cell.parity];
  const found =
    cell.parity === "no-truth" && cell.algorithm_id === "hdbscan"
      ? noise_count > 0
        ? `${noise_count} noise`
        : `k=${n_clusters_found}`
      : `k=${n_clusters_found}${noise_count > 0 ? ` · ${noise_count} noise` : ""}`;
  return `${parity.glyph} ${found} · ${parity.short}`;
}

export function make_grid_ui(): void {
  const container = require_el<HTMLElement>("#demo-grid");
  const status = require_el<HTMLElement>("#grid-status");
  const backend = require_el<HTMLElement>("#grid-backend");
  const footnote = require_el<HTMLElement>("#grid-footnote");

  const views = build_grid(container);
  let datasets: GridDatasets | undefined;

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

  run_grid({
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
      view.annotation.textContent = annotation_text(
        view.cell,
        n_clusters_found,
        noise_count,
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
    },
  });
}
