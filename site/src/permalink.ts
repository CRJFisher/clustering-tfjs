import type { GridAlgorithmId, GridDatasetId } from "./grid_config";

// The shareable permalink: the selected grid cell round-tripped through the URL
// hash, so anyone can reproduce or tweet a specific result. State lives in the
// fragment (never the query) so it never reaches the server and never reloads the
// page; GitHub Pages serves the page statically and ignores the hash entirely.
//
// The schema is versioned: a shared link carries `v=1`, and a future demo that
// changes the encoding bumps the version. An unknown version is dropped wholesale
// on decode rather than guessed at, so an old or future link degrades to defaults
// instead of restoring garbage. Every field is validated too — a bogus enum is
// dropped — and decode NEVER throws, so a hand-mangled hash can only ever lose
// state, never break the page.

export const PERMALINK_VERSION = 1;

// The complete state a "Share this result" click captures: which grid cell is
// selected. Both fields are present on the write side because the live UI always
// has a selected cell.
export interface PermalinkState {
  dataset_id: GridDatasetId;
  algorithm_id: GridAlgorithmId;
}

// The defensive read shape: decode only returns the fields it could trust, so the
// caller keeps its own default for anything absent or rejected.
export interface DecodedPermalink {
  dataset_id?: GridDatasetId;
  algorithm_id?: GridAlgorithmId;
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

export function encode_state(state: PermalinkState): string {
  const params = new URLSearchParams();
  // Deterministic key order (version first) so the same state always stringifies
  // identically — shareable links are stable and diffable.
  params.set("v", String(PERMALINK_VERSION));
  params.set("d", state.dataset_id);
  params.set("a", state.algorithm_id);
  return params.toString();
}

export function decode_state(raw: string): DecodedPermalink {
  const body = raw.startsWith("#") ? raw.slice(1) : raw;
  const params = new URLSearchParams(body);

  // Version gate: an unknown (or missing) schema means the value encoding may
  // differ, so trusting any field would be a guess. Drop everything; the caller
  // keeps all defaults.
  if (Number(params.get("v")) !== PERMALINK_VERSION) return {};

  const result: DecodedPermalink = {};

  const dataset_id = DATASET_IDS.find((id) => id === params.get("d"));
  if (dataset_id !== undefined) result.dataset_id = dataset_id;

  const algorithm_id = ALGORITHM_IDS.find((id) => id === params.get("a"));
  if (algorithm_id !== undefined) result.algorithm_id = algorithm_id;

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
