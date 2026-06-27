import "./style.css";
import { make_race_ui } from "./race_ui";
import { make_grid_ui } from "./grid_ui";
import { make_code_panel } from "./code_panel";
import { wire_copy_button } from "./copy_button";
import { copy_text } from "./clipboard";
import { to_backend_arg } from "./generate_code";
import { build_repo_url, REPO_URL } from "./repo_links";
import { read_url_state, write_url_state } from "./permalink";
import type { PermalinkState } from "./permalink";
import { GRID_CELLS, cell_id_of } from "./grid_config";
import type { GridCell } from "./grid_config";

const INSTALL_CMD = "npm install clustering-tfjs";
// The hero algorithm shown until a cell is chosen or a permalink restores one —
// Spectral on blobs gives the rbf example.
const DEFAULT_CELL_ID = cell_id_of("blobs", "spectral");
const SHARE_RESET_MS = 1400;

const CELLS_BY_ID = new Map<string, GridCell>(
  GRID_CELLS.map((cell) => [cell.cell_id, cell]),
);

function require_el<T extends HTMLElement>(selector: string): T {
  const element = document.querySelector<T>(selector);
  if (!element) throw new Error(`Missing required element ${selector}`);
  return element;
}

function init(): void {
  // Decode once on load (pure, DOM-free). Each restore below applies only the
  // fields that survived validation; everything else keeps its own default.
  const decoded = read_url_state();
  const has_overrides = Object.values(decoded.overrides).some(
    (value) => value !== undefined,
  );

  const race = make_race_ui();
  const grid = make_grid_ui();
  const code_panel = make_code_panel({
    code_el: require_el("#code-panel-code"),
    caption_el: require_el("#code-panel-caption"),
    copy_button: require_el<HTMLButtonElement>("#code-panel-copy"),
    live_el: require_el("#code-panel-live"),
  });

  // Mirror the grid into the code panel: the selected cell, the live params, and
  // the real backend the worker came up on.
  grid.on_selection_change((cell_id) => code_panel.set_selection(cell_id));
  grid.on_overrides_change((overrides) => code_panel.set_overrides(overrides));
  let overrides_restored = false;
  grid.on_backend((actual_backend, live) => {
    if (!live) return;
    const backend = to_backend_arg(actual_backend);
    if (backend) code_panel.set_backend(backend);
    // Override restore waits for a live backend: only then are the controls
    // enabled and the worker warm enough to re-cluster. The latch stops a re-fired
    // on_done (a later worker crash) from re-applying over a dead worker.
    if (has_overrides && !overrides_restored) {
      overrides_restored = true;
      grid.apply_overrides(decoded.overrides);
    }
  });

  // Selection restore is worker-independent — pure DOM state — so it runs at load,
  // focus-free. The code panel renders the initial cell immediately.
  const initial_cell =
    decoded.dataset_id && decoded.algorithm_id
      ? cell_id_of(decoded.dataset_id, decoded.algorithm_id)
      : DEFAULT_CELL_ID;
  grid.select_cell(initial_cell);
  code_panel.set_selection(initial_cell);

  // Race restore: set n and run exactly once (the race never auto-runs otherwise).
  if (decoded.n !== undefined) {
    race.set_n(decoded.n);
    race.run();
  }

  // Conversion surfaces: install copy buttons, UTM-tagged star/repo links.
  wire_copy_button(
    require_el<HTMLButtonElement>("#hero-install-copy"),
    require_el("#hero-cta-live"),
    () => INSTALL_CMD,
    "npm command",
  );
  wire_copy_button(
    require_el<HTMLButtonElement>("#code-install-copy"),
    require_el("#code-cta-live"),
    () => INSTALL_CMD,
    "npm command",
  );

  require_el<HTMLAnchorElement>("#hero-star-link").href = build_repo_url(
    REPO_URL,
    "header",
  );
  require_el<HTMLAnchorElement>("#code-star-link").href = build_repo_url(
    REPO_URL,
    "code_panel",
  );
  require_el<HTMLAnchorElement>(".repo-link").href = build_repo_url(
    REPO_URL,
    "footer",
  );

  // Share: gather the live state, write the hash in place (replaceState — no
  // reload, no history entry, no re-restore), and copy the full URL. Clipboard is
  // reached only here, behind the click gesture.
  const share_button = require_el<HTMLButtonElement>("#code-share-btn");
  const share_live = require_el("#code-cta-live");
  let share_reset: ReturnType<typeof setTimeout> | undefined;
  share_button.addEventListener("click", () => {
    const selected = grid.get_selected_cell();
    const cell = selected ? CELLS_BY_ID.get(selected) : undefined;
    const state: PermalinkState = {
      n: race.get_n(),
      dataset_id: cell ? cell.dataset_id : "blobs",
      algorithm_id: cell ? cell.algorithm_id : "spectral",
      overrides: grid.get_overrides(),
    };
    write_url_state(state);
    void copy_text(location.href).then((ok) => {
      share_live.textContent = ok
        ? "Shareable link copied"
        : "Link is in the address bar — copy it manually.";
      if (share_reset !== undefined) clearTimeout(share_reset);
      share_reset = setTimeout(() => {
        share_live.textContent = "";
      }, SHARE_RESET_MS);
    });
  });
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", init);
} else {
  init();
}
