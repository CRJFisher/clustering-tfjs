import {
  NUMERIC_CONTROL_BOUNDS,
  any_override_active,
  clamp_numeric,
} from "./grid_controls";
import type {
  AgglomerativeLinkage,
  ControlOverrides,
} from "./grid_controls";

// Builds the live parameter control strip and owns the ControlOverrides it edits.
// Each control starts in Auto (the override field undefined → grid_config's curated
// per-dataset value); moving it off Auto sets the field, and the panel calls
// `on_change` so the orchestrator can diff and re-cluster. The panel knows nothing
// about the grid or the worker — it only edits overrides and announces changes.

type NumericControlId = keyof typeof NUMERIC_CONTROL_BOUNDS;

interface NumericSpec {
  id: NumericControlId;
  label: string;
  caption: string;
  // Decimal places for the readout: gamma is continuous, the rest are integers.
  decimals: number;
}

const NUMERIC_SPECS: NumericSpec[] = [
  {
    id: "n_clusters",
    label: "Clusters (k)",
    caption:
      "The number of clusters K-Means, Spectral, Agglomerative and SOM are forced " +
      "to find. HDBSCAN ignores it — it discovers the count from density.",
    decimals: 0,
  },
  {
    id: "spectral_gamma",
    label: "Spectral · RBF γ",
    caption:
      "RBF kernel width: a higher γ makes similarity fall off faster, so only very " +
      "close points group together. Only affects cells using the RBF affinity — " +
      "switch the affinity control to apply it across the column.",
    decimals: 1,
  },
  {
    id: "hdbscan_min_cluster_size",
    label: "HDBSCAN · min cluster size",
    caption:
      "The smallest group HDBSCAN will call a cluster; anything sparser is labelled " +
      "noise. Raise it to dissolve specks into noise.",
    decimals: 0,
  },
  {
    id: "som_grid_size",
    label: "SOM · grid size",
    caption:
      "The width and height of the self-organizing map's neuron lattice; a larger " +
      "map captures finer structure before it is merged down to k clusters.",
    decimals: 0,
  },
];

const AFFINITY_CAPTION =
  "How Spectral measures similarity: nearest-neighbors follows curved manifolds; " +
  "the RBF kernel suits compact blobs.";
const LINKAGE_CAPTION =
  "Which distance fuses clusters when merging: Ward favours compact balls; single " +
  "linkage chains along thin manifolds.";

export interface GridControlsPanel {
  get_overrides: () => ControlOverrides;
  set_enabled: (enabled: boolean) => void;
  // Drive every control to match the given overrides (a restored permalink),
  // re-clustering once. The programmatic inverse of the user moving the controls.
  apply_overrides: (overrides: ControlOverrides) => void;
}

// One numeric control's interactive elements plus an `apply` that sets it to a
// target value (or back to Auto) WITHOUT firing on_change — kept so enable/disable,
// reset, and a restored permalink can all drive it without re-querying the DOM and
// without fighting the per-control input listeners.
interface NumericHandle {
  id: NumericControlId;
  auto: HTMLInputElement;
  slider: HTMLInputElement;
  value: HTMLElement;
  apply: (next: number | undefined) => void;
}

// One select's element, its typed string→override mapper, and how to read its
// target value out of an overrides record — so apply_overrides can set both
// selects through the same narrowing the change listeners use.
interface SelectHandle {
  select: HTMLSelectElement;
  set: (raw: string) => void;
  read_target: (overrides: ControlOverrides) => string;
}

const ALL_OVERRIDE_KEYS: (keyof ControlOverrides)[] = [
  "n_clusters",
  "spectral_affinity",
  "spectral_gamma",
  "hdbscan_min_cluster_size",
  "agglomerative_linkage",
  "som_grid_size",
];

function format_value(value: number, decimals: number): string {
  return decimals === 0 ? String(value) : value.toFixed(decimals);
}

export function make_grid_controls(
  mount: HTMLElement,
  on_change: () => void,
): GridControlsPanel {
  const overrides: ControlOverrides = {};
  const numeric_handles: NumericHandle[] = [];
  const select_handles: SelectHandle[] = [];
  let enabled = false;

  // The panel starts inert until the grid's worker backend is warm; announce that
  // busy state so the disabled controls aren't a silent dead zone for AT.
  mount.setAttribute("aria-busy", "true");

  const reset_button = document.createElement("button");
  reset_button.type = "button";
  reset_button.className = "grid-controls__reset";
  // "Auto" is the curated per-dataset grid, NOT scikit-learn's library defaults —
  // name what reset actually restores (the verified grid), matching the intro copy.
  reset_button.textContent = "Reset to the verified grid";
  reset_button.disabled = true;

  // The reset button is live only when something is off Auto and the panel is
  // enabled; sliders are live only when enabled AND their Auto box is unchecked.
  function refresh_disabled(): void {
    reset_button.disabled = !enabled || !any_override_active(overrides);
    for (const handle of numeric_handles) {
      handle.auto.disabled = !enabled;
      handle.slider.disabled = !enabled || handle.auto.checked;
    }
    for (const handle of select_handles) handle.select.disabled = !enabled;
  }

  function changed(): void {
    refresh_disabled();
    on_change();
  }

  function build_numeric(spec: NumericSpec): HTMLElement {
    const bounds = NUMERIC_CONTROL_BOUNDS[spec.id];
    const slider_id = `ctl-${spec.id}`;
    const caption_id = `${slider_id}-caption`;

    const control = document.createElement("div");
    control.className = "grid-controls__control";

    const row = document.createElement("div");
    row.className = "grid-controls__row";
    const label = document.createElement("label");
    label.className = "grid-controls__label";
    label.htmlFor = slider_id;
    label.textContent = spec.label;
    const value = document.createElement("span");
    value.className = "grid-controls__value";
    value.textContent = "Auto";
    row.append(label, value);

    const widget = document.createElement("div");
    widget.className = "grid-controls__widget";

    const auto_label = document.createElement("label");
    auto_label.className = "grid-controls__auto";
    const auto = document.createElement("input");
    auto.type = "checkbox";
    auto.checked = true;
    // The bare " Auto" text reads without context when tabbed to; name the
    // parameter so a screen reader announces which control this Auto governs.
    auto.setAttribute("aria-label", `Auto for ${spec.label}`);
    auto_label.append(auto, document.createTextNode(" Auto"));

    const slider = document.createElement("input");
    slider.type = "range";
    slider.id = slider_id;
    slider.className = "grid-controls__slider";
    slider.min = String(bounds.min);
    slider.max = String(bounds.max);
    slider.step = String(bounds.step);
    slider.value = String(bounds.default_value);
    slider.disabled = true;
    slider.setAttribute("aria-describedby", caption_id);
    slider.setAttribute("aria-valuetext", "Auto (per-dataset)");

    widget.append(auto_label, slider);

    const caption = document.createElement("p");
    caption.className = "grid-controls__caption";
    caption.id = caption_id;
    caption.textContent = spec.caption;

    control.append(row, widget, caption);

    function read(): number {
      return clamp_numeric(spec.id, Number(slider.value));
    }

    function show_override(n: number): void {
      const text = format_value(n, spec.decimals);
      value.textContent = text;
      slider.setAttribute("aria-valuetext", text);
    }

    auto.addEventListener("change", () => {
      if (auto.checked) {
        overrides[spec.id] = undefined;
        value.textContent = "Auto";
        slider.setAttribute("aria-valuetext", "Auto (per-dataset)");
      } else {
        const n = read();
        overrides[spec.id] = n;
        show_override(n);
      }
      changed();
    });

    slider.addEventListener("input", () => {
      // Dragging implies leaving Auto: untick it so the readout and the grid agree.
      if (auto.checked) auto.checked = false;
      const n = read();
      overrides[spec.id] = n;
      show_override(n);
      changed();
    });

    // Set the control to a target value (or back to Auto when undefined) writing
    // exactly the state the listeners above write, but WITHOUT calling changed() —
    // the batch caller fires one changed() for the whole apply.
    function apply(next: number | undefined): void {
      if (next === undefined) {
        auto.checked = true;
        overrides[spec.id] = undefined;
        value.textContent = "Auto";
        slider.setAttribute("aria-valuetext", "Auto (per-dataset)");
      } else {
        const n = clamp_numeric(spec.id, next);
        auto.checked = false;
        slider.value = String(n);
        overrides[spec.id] = n;
        show_override(n);
      }
    }

    numeric_handles.push({ id: spec.id, auto, slider, value, apply });
    return control;
  }

  function build_select(
    id: string,
    label_text: string,
    caption_text: string,
    options: { value: string; label: string }[],
    on_select: (raw: string) => void,
    read_target: (overrides: ControlOverrides) => string,
  ): HTMLElement {
    const select_id = `ctl-${id}`;
    const caption_id = `${select_id}-caption`;

    const control = document.createElement("div");
    control.className = "grid-controls__control";

    const row = document.createElement("div");
    row.className = "grid-controls__row";
    const label = document.createElement("label");
    label.className = "grid-controls__label";
    label.htmlFor = select_id;
    label.textContent = label_text;
    row.append(label);

    const widget = document.createElement("div");
    widget.className = "grid-controls__widget";
    const select = document.createElement("select");
    select.id = select_id;
    select.className = "grid-controls__select";
    select.disabled = true;
    select.setAttribute("aria-describedby", caption_id);
    const auto_option = document.createElement("option");
    auto_option.value = "";
    auto_option.textContent = "Auto (per-dataset)";
    select.append(auto_option);
    for (const option of options) {
      const element = document.createElement("option");
      element.value = option.value;
      element.textContent = option.label;
      select.append(element);
    }
    widget.append(select);

    const caption = document.createElement("p");
    caption.className = "grid-controls__caption";
    caption.id = caption_id;
    caption.textContent = caption_text;

    control.append(row, widget, caption);

    select.addEventListener("change", () => {
      on_select(select.value);
      changed();
    });

    select_handles.push({ select, set: on_select, read_target });
    return control;
  }

  // Explicit, typed string→union narrowing keeps the override fields strongly
  // typed with no casts; an empty value is the Auto sentinel.
  function set_affinity(raw: string): void {
    overrides.spectral_affinity =
      raw === "rbf"
        ? "rbf"
        : raw === "nearest_neighbors"
          ? "nearest_neighbors"
          : undefined;
  }

  function set_linkage(raw: string): void {
    const linkages: AgglomerativeLinkage[] = [
      "ward",
      "complete",
      "average",
      "single",
    ];
    overrides.agglomerative_linkage = linkages.find((l) => l === raw);
  }

  // Build order groups the global k control first, then the per-algorithm knobs.
  mount.append(build_numeric(NUMERIC_SPECS[0]));
  mount.append(
    build_select(
      "spectral_affinity",
      "Spectral · affinity",
      AFFINITY_CAPTION,
      [
        { value: "nearest_neighbors", label: "Nearest neighbors" },
        { value: "rbf", label: "RBF kernel" },
      ],
      set_affinity,
      (o) => o.spectral_affinity ?? "",
    ),
  );
  mount.append(build_numeric(NUMERIC_SPECS[1]));
  mount.append(
    build_select(
      "agglomerative_linkage",
      "Agglomerative · linkage",
      LINKAGE_CAPTION,
      [
        { value: "ward", label: "Ward" },
        { value: "complete", label: "Complete" },
        { value: "average", label: "Average" },
        { value: "single", label: "Single" },
      ],
      set_linkage,
      (o) => o.agglomerative_linkage ?? "",
    ),
  );
  mount.append(build_numeric(NUMERIC_SPECS[2]));
  mount.append(build_numeric(NUMERIC_SPECS[3]));

  reset_button.addEventListener("click", () => {
    for (const key of ALL_OVERRIDE_KEYS) overrides[key] = undefined;
    for (const handle of numeric_handles) {
      handle.auto.checked = true;
      handle.value.textContent = "Auto";
      handle.slider.setAttribute("aria-valuetext", "Auto (per-dataset)");
    }
    for (const handle of select_handles) handle.select.value = "";
    changed();
  });

  mount.append(reset_button);

  return {
    get_overrides: () => overrides,
    set_enabled: (next: boolean) => {
      enabled = next;
      mount.setAttribute("aria-busy", next ? "false" : "true");
      refresh_disabled();
    },
    apply_overrides: (next: ControlOverrides) => {
      for (const handle of numeric_handles) handle.apply(next[handle.id]);
      for (const handle of select_handles) {
        const raw = handle.read_target(next);
        handle.select.value = raw;
        handle.set(raw);
      }
      // One changed() for the whole batch: a single refresh_disabled() + a single
      // on_change(), so the restore re-clusters once like any settled user move.
      changed();
    },
  };
}
