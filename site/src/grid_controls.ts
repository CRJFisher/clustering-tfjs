import { GRID_CELLS } from "./grid_config";
import type {
  GridAlgorithmId,
  GridCell,
  GridParams,
} from "./grid_config";

// Resolves the live parameter controls against the grid's curated per-cell params.
//
// The 5×5 grid (grid_config) curates DIFFERENT params per (algorithm, dataset) so
// each algorithm is shown at its best: Spectral uses nearest_neighbors on the
// curved rows but rbf on blobs, Agglomerative uses single linkage on the thin
// manifolds but ward on blobs, n_clusters is 2 for the two-shape rows and 3 for
// the rest. A single global slider cannot express that, so every control has an
// AUTO state meaning "use the curated per-dataset value" — Auto reproduces the
// exact grid_config grid, badges and all.
//
// The instant a control leaves Auto (an explicit override), that parameter is
// applied globally to its column(s). A cell whose effective params then differ
// from its curated params is "exploratory": its scikit-learn parity claim no
// longer holds (the override combo was never checked against scikit-learn), so the
// UI must drop the parity badge for that cell. Resetting every control to Auto
// restores the curated params and the badges. This module is the single source of
// truth for both halves — what to run, and whether a cell may still claim parity.

export type SpectralAffinity = "nearest_neighbors" | "rbf";
export type AgglomerativeLinkage = "ward" | "complete" | "average" | "single";

// All fields optional; `undefined` means Auto (defer to the curated per-dataset
// value). A set field overrides that parameter across every cell its control
// touches.
export interface ControlOverrides {
  n_clusters?: number;
  spectral_affinity?: SpectralAffinity;
  spectral_gamma?: number;
  hdbscan_min_cluster_size?: number;
  agglomerative_linkage?: AgglomerativeLinkage;
  som_grid_size?: number;
}

export type ControlId = keyof ControlOverrides;

// Which algorithm columns each control re-clusters when moved off Auto. n_clusters
// drives all four partitioning columns; HDBSCAN discovers its own count from
// density, so it is deliberately absent.
export const CONTROL_ALGORITHMS: Record<ControlId, GridAlgorithmId[]> = {
  n_clusters: ["kmeans", "spectral", "agglomerative", "som"],
  spectral_affinity: ["spectral"],
  spectral_gamma: ["spectral"],
  hdbscan_min_cluster_size: ["hdbscan"],
  agglomerative_linkage: ["agglomerative"],
  som_grid_size: ["som"],
};

export interface NumericControlBounds {
  min: number;
  max: number;
  step: number;
  // The value the slider jumps to the moment it leaves Auto — the library's
  // typical default for the parameter, so the first override is a sensible one.
  default_value: number;
}

export const NUMERIC_CONTROL_BOUNDS: Record<
  "n_clusters" | "spectral_gamma" | "hdbscan_min_cluster_size" | "som_grid_size",
  NumericControlBounds
> = {
  n_clusters: { min: 2, max: 6, step: 1, default_value: 3 },
  spectral_gamma: { min: 0.1, max: 10, step: 0.1, default_value: 1.0 },
  hdbscan_min_cluster_size: { min: 5, max: 60, step: 5, default_value: 15 },
  som_grid_size: { min: 4, max: 12, step: 1, default_value: 6 },
};

// Library defaults used only when a control switches a cell INTO an affinity its
// curated params did not carry (e.g. forcing rbf onto a curated nearest_neighbors
// cell, which has no gamma). They mirror grid_config's own rbf/kNN choices so the
// switched cell starts from the same place the curated grid would.
const DEFAULT_GAMMA = 1.0;
const DEFAULT_N_NEIGHBORS = 10;

export function clamp_numeric(
  control: keyof typeof NUMERIC_CONTROL_BOUNDS,
  value: number,
): number {
  const { min, max } = NUMERIC_CONTROL_BOUNDS[control];
  return Math.min(max, Math.max(min, value));
}

// Curated params + overrides → the params this cell is actually fit with. With no
// overrides this returns the curated params unchanged, so the Auto grid is
// byte-identical to grid_config's.
export function resolve_params(
  cell: GridCell,
  overrides: ControlOverrides,
): GridParams {
  const curated = cell.params;
  switch (curated.algorithm_id) {
    case "kmeans":
      return { ...curated, n_clusters: overrides.n_clusters ?? curated.n_clusters };
    case "spectral": {
      const n_clusters = overrides.n_clusters ?? curated.n_clusters;
      const affinity = overrides.spectral_affinity ?? curated.affinity;
      if (affinity === "rbf") {
        return {
          algorithm_id: "spectral",
          n_clusters,
          affinity: "rbf",
          gamma:
            overrides.spectral_gamma ??
            (curated.affinity === "rbf" ? curated.gamma : DEFAULT_GAMMA),
          n_init: curated.n_init,
          random_state: curated.random_state,
        };
      }
      return {
        algorithm_id: "spectral",
        n_clusters,
        affinity: "nearest_neighbors",
        n_neighbors:
          curated.affinity === "nearest_neighbors"
            ? curated.n_neighbors
            : DEFAULT_N_NEIGHBORS,
        n_init: curated.n_init,
        random_state: curated.random_state,
      };
    }
    case "agglomerative":
      return {
        ...curated,
        n_clusters: overrides.n_clusters ?? curated.n_clusters,
        linkage: overrides.agglomerative_linkage ?? curated.linkage,
      };
    case "hdbscan":
      return {
        ...curated,
        min_cluster_size:
          overrides.hdbscan_min_cluster_size ?? curated.min_cluster_size,
      };
    case "som": {
      const size = overrides.som_grid_size;
      return {
        ...curated,
        n_clusters: overrides.n_clusters ?? curated.n_clusters,
        grid_width: size ?? curated.grid_width,
        grid_height: size ?? curated.grid_height,
      };
    }
  }
}

// Structural equality over the flat param records, switched per algorithm so the
// comparison is fully typed with no casts. Two params are equal iff every field of
// their shared shape matches. The orchestrator diffs successive resolved params
// with this to decide which cells actually changed and need re-fitting.
export function params_equal(a: GridParams, b: GridParams): boolean {
  switch (a.algorithm_id) {
    case "kmeans":
      return (
        b.algorithm_id === "kmeans" &&
        a.n_clusters === b.n_clusters &&
        a.n_init === b.n_init &&
        a.random_state === b.random_state
      );
    case "spectral":
      return (
        b.algorithm_id === "spectral" &&
        a.n_clusters === b.n_clusters &&
        a.affinity === b.affinity &&
        a.gamma === b.gamma &&
        a.n_neighbors === b.n_neighbors &&
        a.n_init === b.n_init &&
        a.random_state === b.random_state
      );
    case "agglomerative":
      return (
        b.algorithm_id === "agglomerative" &&
        a.n_clusters === b.n_clusters &&
        a.linkage === b.linkage
      );
    case "hdbscan":
      return (
        b.algorithm_id === "hdbscan" &&
        a.min_cluster_size === b.min_cluster_size &&
        a.min_samples === b.min_samples
      );
    case "som":
      return (
        b.algorithm_id === "som" &&
        a.grid_width === b.grid_width &&
        a.grid_height === b.grid_height &&
        a.num_epochs === b.num_epochs &&
        a.random_state === b.random_state &&
        a.n_clusters === b.n_clusters &&
        a.cluster_linkage === b.cluster_linkage
      );
  }
}

// A cell is exploratory — and must drop its scikit-learn parity badge — exactly
// when its effective params diverge from curated. Defining this off the resolved
// params (not "is any control affecting this column active?") keeps it honest in
// both directions: a gamma override left inert on a nearest_neighbors cell does
// not change that cell, so the cell keeps its badge.
export function is_overridden(
  cell: GridCell,
  overrides: ControlOverrides,
): boolean {
  return !params_equal(resolve_params(cell, overrides), cell.params);
}

export interface ResolvedCell {
  cell: GridCell;
  params: GridParams;
  overridden: boolean;
}

// Every cell's resolved params and parity state under the current overrides, in
// the canonical grid order. The orchestrator diffs successive snapshots to decide
// which cells actually need re-fitting.
export function resolve_all(overrides: ControlOverrides): ResolvedCell[] {
  return GRID_CELLS.map((cell) => ({
    cell,
    params: resolve_params(cell, overrides),
    overridden: is_overridden(cell, overrides),
  }));
}

// True when any control is off Auto — drives the "Reset to scikit-learn defaults"
// affordance.
export function any_override_active(overrides: ControlOverrides): boolean {
  return Object.values(overrides).some((value) => value !== undefined);
}
