import "./style.css";
import { make_benchmark_ui } from "./benchmark_ui";
import { make_grid_ui } from "./grid_ui";
import { make_code_panel } from "./code_panel";
import { wire_copy_button } from "./copy_button";
import { copy_text } from "./clipboard";
import { to_backend_arg } from "./generate_code";
import { build_repo_url, REPO_URL } from "./repo_links";
import { read_url_state, write_url_state } from "./permalink";
import type { PermalinkState } from "./permalink";
import { cell_id_of, get_grid_cell } from "./grid_config";

const INSTALL_CMD = "npm install clustering-tfjs";
// The hero algorithm shown until a cell is chosen or a permalink restores one —
// Spectral on blobs gives the rbf example.
const DEFAULT_CELL_ID = cell_id_of("blobs", "spectral");
const SHARE_RESET_MS = 1400;

function require_el<T extends HTMLElement>(selector: string): T {
  const element = document.querySelector<T>(selector);
  if (!element) throw new Error(`Missing required element ${selector}`);
  return element;
}

function init(): void {
  // Decode once on load (pure, DOM-free). The restore below applies only the
  // fields that survived validation; everything else keeps its own default.
  const decoded = read_url_state();

  // Self-wires its own Run button; no handle needed on the main thread.
  make_benchmark_ui();
  const grid = make_grid_ui();
  const code_panel = make_code_panel({
    code_el: require_el("#code-panel-code"),
    caption_el: require_el("#code-panel-caption"),
    copy_button: require_el<HTMLButtonElement>("#code-panel-copy"),
    live_el: require_el("#code-panel-live"),
  });

  // Mirror the grid into the code panel: the selected cell and the real backend
  // the worker came up on.
  grid.on_selection_change((cell_id) => code_panel.set_selection(cell_id));
  grid.on_backend((actual_backend, live) => {
    if (!live) return;
    const backend = to_backend_arg(actual_backend);
    if (backend) code_panel.set_backend(backend);
  });

  // Selection restore is worker-independent — pure DOM state — so it runs at load,
  // focus-free. The code panel renders the initial cell immediately.
  const initial_cell =
    decoded.dataset_id && decoded.algorithm_id
      ? cell_id_of(decoded.dataset_id, decoded.algorithm_id)
      : DEFAULT_CELL_ID;
  grid.select_cell(initial_cell);
  code_panel.set_selection(initial_cell);

  // The grid runs only on explicit click — clustering 25 cells costs real compute,
  // so the page never spends it unasked. (The benchmark UI self-wires its own Run
  // button.) Disable the button once used so a second click can't re-trigger it.
  const populate_button = require_el<HTMLButtonElement>("#populate-grid");
  populate_button.addEventListener("click", () => {
    populate_button.disabled = true;
    grid.populate();
  });

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
  // Its own live region, not the install-copy one: a Share right after a Copy must
  // not clobber the copy announcement (or be clobbered by it).
  const share_live = require_el("#code-share-live");
  let share_reset: ReturnType<typeof setTimeout> | undefined;
  share_button.addEventListener("click", () => {
    const selected = grid.get_selected_cell();
    const cell = selected ? get_grid_cell(selected) : undefined;
    const state: PermalinkState = {
      dataset_id: cell ? cell.dataset_id : "blobs",
      algorithm_id: cell ? cell.algorithm_id : "spectral",
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
