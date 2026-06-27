import {
  GRID_ALGORITHMS,
  GRID_CELLS,
  GRID_DATASETS,
  get_grid_cell,
} from "./grid_config";
import { resolve_params } from "./grid_controls";
import type { ControlOverrides } from "./grid_controls";
import {
  DEFAULT_BACKEND_ARG,
  generate_code,
} from "./generate_code";
import type { BackendArg } from "./generate_code";
import { wire_copy_button } from "./copy_button";

// The conversion code panel: it mirrors the selected grid cell as the real ~5
// lines a visitor would write, regenerated whenever the selection, the live
// parameter overrides, or the worker's backend change. The code comes straight
// from the same resolve_params the grid fits with, so what the panel shows can
// never drift from what the cell actually computed.

export interface CodePanelElements {
  code_el: HTMLElement;
  caption_el: HTMLElement;
  copy_button: HTMLButtonElement;
  live_el: HTMLElement;
}

export interface CodePanel {
  set_selection: (cell_id: string) => void;
  set_overrides: (overrides: ControlOverrides) => void;
  set_backend: (backend: BackendArg) => void;
  get_code: () => string;
}

const DATASET_LABELS = new Map(
  GRID_DATASETS.map((dataset) => [dataset.id, dataset.label]),
);
const ALGORITHM_LABELS = new Map(
  GRID_ALGORITHMS.map((algorithm) => [algorithm.id, algorithm.label]),
);

export function make_code_panel(elements: CodePanelElements): CodePanel {
  let current_cell = GRID_CELLS[0];
  let current_overrides: ControlOverrides = {};
  let current_backend: BackendArg = DEFAULT_BACKEND_ARG;
  let current_code = "";

  function render(): void {
    current_code = generate_code({
      params: resolve_params(current_cell, current_overrides),
      backend: current_backend,
    });
    elements.code_el.textContent = current_code;
    elements.caption_el.textContent =
      `${DATASET_LABELS.get(current_cell.dataset_id) ?? current_cell.dataset_id}` +
      ` · ${ALGORITHM_LABELS.get(current_cell.algorithm_id) ?? current_cell.algorithm_id}`;
  }

  wire_copy_button(
    elements.copy_button,
    elements.live_el,
    () => current_code,
    "Code",
  );

  render();

  return {
    set_selection: (cell_id: string): void => {
      const cell = get_grid_cell(cell_id);
      if (!cell) return;
      current_cell = cell;
      render();
    },
    set_overrides: (overrides: ControlOverrides): void => {
      current_overrides = overrides;
      render();
    },
    set_backend: (backend: BackendArg): void => {
      current_backend = backend;
      render();
    },
    get_code: () => current_code,
  };
}
