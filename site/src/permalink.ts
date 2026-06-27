import { N_MAX, N_MIN } from "./crossover";
import { NUMERIC_CONTROL_BOUNDS, clamp_numeric } from "./grid_controls";
import type {
  AgglomerativeLinkage,
  ControlOverrides,
} from "./grid_controls";
import type { GridAlgorithmId, GridDatasetId } from "./grid_config";

// The shareable permalink: the demo's tweakable state round-tripped through the
// URL hash, so anyone can reproduce or tweet a specific result. State lives in the
// fragment (never the query) so it never reaches the server and never reloads the
// page; GitHub Pages serves the page statically and ignores the hash entirely.
//
// The schema is versioned: a shared link carries `v=1`, and a future demo that
// changes the encoding bumps the version. An unknown version is dropped wholesale
// on decode rather than guessed at, so an old or future link degrades to defaults
// instead of restoring garbage. Every individual field is validated too — an
// out-of-range number clamps to its control's bounds, a bogus enum is dropped —
// and decode NEVER throws, so a hand-mangled hash can only ever lose state, never
// break the page.

export const PERMALINK_VERSION = 1;

// The complete state a "Share this result" click captures. Every field is present
// on the write side because the live UI always has all of it.
export interface PermalinkState {
  n: number;
  dataset_id: GridDatasetId;
  algorithm_id: GridAlgorithmId;
  overrides: ControlOverrides;
}

// The defensive read shape: decode only returns the fields it could trust, so the
// caller keeps its own default for anything absent or rejected. `overrides` is
// always present (`{}` when nothing valid decoded), mirroring ControlOverrides'
// all-optional shape.
export interface DecodedPermalink {
  n?: number;
  dataset_id?: GridDatasetId;
  algorithm_id?: GridAlgorithmId;
  overrides: ControlOverrides;
}

const DATASET_IDS: GridDatasetId[] = [
  "moons",
  "circles",
  "blobs",
  "aniso",
  "none",
];
const ALGORITHM_IDS: GridAlgorithmId[] = [
  "kmeans",
  "spectral",
  "agglomerative",
  "hdbscan",
  "som",
];
const LINKAGES: AgglomerativeLinkage[] = [
  "ward",
  "complete",
  "average",
  "single",
];

export function encode_state(state: PermalinkState): string {
  const params = new URLSearchParams();
  // Deterministic key order (version first) so the same state always stringifies
  // identically — shareable links are stable and diffable.
  params.set("v", String(PERMALINK_VERSION));
  params.set("n", String(state.n));
  params.set("d", state.dataset_id);
  params.set("a", state.algorithm_id);

  const o = state.overrides;
  // An Auto control (undefined) contributes no key, so a default-state link is
  // just `v&n&d&a` — short, and forward-compatible with new controls.
  if (o.n_clusters !== undefined) params.set("k", String(o.n_clusters));
  if (o.spectral_affinity !== undefined) {
    params.set("sa", o.spectral_affinity === "rbf" ? "rbf" : "nn");
  }
  if (o.spectral_gamma !== undefined) params.set("sg", String(o.spectral_gamma));
  if (o.hdbscan_min_cluster_size !== undefined) {
    params.set("hm", String(o.hdbscan_min_cluster_size));
  }
  if (o.agglomerative_linkage !== undefined) {
    params.set("al", o.agglomerative_linkage);
  }
  if (o.som_grid_size !== undefined) params.set("som", String(o.som_grid_size));

  return params.toString();
}

// Out-of-range → clamp to the nearest bound (an extreme but meaningful intent);
// non-numeric → undefined (drop, keep the caller's default). The two failure modes
// are deliberately distinct.
function decode_clamped_int(
  raw: string | null,
  min: number,
  max: number,
): number | undefined {
  if (raw === null) return undefined;
  const value = Number(raw);
  if (Number.isNaN(value)) return undefined;
  return Math.min(max, Math.max(min, Math.round(value)));
}

function decode_override(
  raw: string | null,
  control: keyof typeof NUMERIC_CONTROL_BOUNDS,
  integer: boolean,
): number | undefined {
  if (raw === null) return undefined;
  const value = Number(raw);
  if (Number.isNaN(value)) return undefined;
  return clamp_numeric(control, integer ? Math.round(value) : value);
}

export function decode_state(raw: string): DecodedPermalink {
  const body = raw.startsWith("#") ? raw.slice(1) : raw;
  const params = new URLSearchParams(body);

  // Version gate: an unknown (or missing) schema means the value encoding may
  // differ, so trusting any field would be a guess. Drop everything; the caller
  // keeps all defaults.
  if (Number(params.get("v")) !== PERMALINK_VERSION) return { overrides: {} };

  const result: DecodedPermalink = { overrides: {} };

  const n = decode_clamped_int(params.get("n"), N_MIN, N_MAX);
  if (n !== undefined) result.n = n;

  const dataset_id = DATASET_IDS.find((id) => id === params.get("d"));
  if (dataset_id !== undefined) result.dataset_id = dataset_id;

  const algorithm_id = ALGORITHM_IDS.find((id) => id === params.get("a"));
  if (algorithm_id !== undefined) result.algorithm_id = algorithm_id;

  const overrides: ControlOverrides = {};

  const k = decode_override(params.get("k"), "n_clusters", true);
  if (k !== undefined) overrides.n_clusters = k;

  const sa = params.get("sa");
  if (sa === "rbf") overrides.spectral_affinity = "rbf";
  else if (sa === "nn") overrides.spectral_affinity = "nearest_neighbors";

  const sg = decode_override(params.get("sg"), "spectral_gamma", false);
  if (sg !== undefined) overrides.spectral_gamma = sg;

  const hm = decode_override(
    params.get("hm"),
    "hdbscan_min_cluster_size",
    true,
  );
  if (hm !== undefined) overrides.hdbscan_min_cluster_size = hm;

  const al = LINKAGES.find((l) => l === params.get("al"));
  if (al !== undefined) overrides.agglomerative_linkage = al;

  const som = decode_override(params.get("som"), "som_grid_size", true);
  if (som !== undefined) overrides.som_grid_size = som;

  result.overrides = overrides;
  return result;
}

// The two adapters are the only code that touches `location`; all logic lives in
// the pure pair above (unit-tested in the DOM-free node env).
export function read_url_state(): DecodedPermalink {
  return decode_state(location.hash);
}

// Write via replaceState, not `location.hash =`: it updates the address bar with
// neither a reload, a history entry, nor a `hashchange` event — so sharing can
// never retrigger the load-time restore.
export function write_url_state(state: PermalinkState): void {
  history.replaceState(null, "", `#${encode_state(state)}`);
}
