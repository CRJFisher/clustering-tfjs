import type { ToyDatasetId } from "./make_toy_datasets";

// The single source of truth for the parity grid: the five datasets (rows), the
// five algorithms (columns), the exact per-cell params the worker constructs each
// estimator with, and the parity tier each cell honestly advertises. Both the
// worker (what to run) and the UI (headers, render targets, annotations) read
// this, so the rendered grid can never drift from what was computed.

export type GridDatasetId = ToyDatasetId;

export type GridAlgorithmId =
  | "kmeans"
  | "spectral"
  | "agglomerative"
  | "hdbscan"
  | "som";

export interface GridDatasetSpec {
  id: GridDatasetId;
  label: string;
}

export interface GridAlgorithmSpec {
  id: GridAlgorithmId;
  label: string;
}

// Row order top-to-bottom and column order left-to-right of the rendered grid.
export const GRID_DATASETS: GridDatasetSpec[] = [
  { id: "moons", label: "Two moons" },
  { id: "circles", label: "Concentric circles" },
  { id: "blobs", label: "Blobs" },
  { id: "aniso", label: "Anisotropic" },
  { id: "none", label: "No structure" },
];

export const GRID_ALGORITHMS: GridAlgorithmSpec[] = [
  { id: "kmeans", label: "K-Means" },
  { id: "spectral", label: "Spectral" },
  { id: "agglomerative", label: "Agglomerative" },
  { id: "hdbscan", label: "HDBSCAN" },
  { id: "som", label: "SOM" },
];

// Per-algorithm params, discriminated by algorithm_id so the worker's estimator
// factory is exhaustively typed with no casts. SOM is two-phase (fit then
// cluster), so it carries both the training grid and the macro-cluster target.
export type GridParams =
  | {
      algorithm_id: "kmeans";
      n_clusters: number;
      n_init: number;
      random_state: number;
    }
  | {
      algorithm_id: "spectral";
      n_clusters: number;
      affinity: "rbf" | "nearest_neighbors";
      gamma?: number;
      n_neighbors?: number;
      n_init: number;
      random_state: number;
    }
  | {
      algorithm_id: "agglomerative";
      n_clusters: number;
      linkage: "ward" | "complete" | "average" | "single";
    }
  | {
      algorithm_id: "hdbscan";
      min_cluster_size: number;
      min_samples: number;
    }
  | {
      algorithm_id: "som";
      grid_width: number;
      grid_height: number;
      num_epochs: number;
      random_state: number;
      n_clusters: number;
      cluster_linkage: "ward" | "average";
    };

// The four honest parity tiers a cell can claim, mirroring the race fold's
// precise-over-salesy labelling:
// - matches: the library's float32 labels reproduce scikit-learn's result.
// - drifts: cluster cores match scikit-learn, but a few boundary points move
//   under float32 (Spectral's eigen-embedding, HDBSCAN's density tree).
// - no-reference: SOM has no scikit-learn counterpart in this comparison.
// - no-truth: uniform data has no real clusters to recover.
export type ParityTier = "matches" | "drifts" | "no-reference" | "no-truth";

export interface GridCell {
  cell_id: string;
  dataset_id: GridDatasetId;
  algorithm_id: GridAlgorithmId;
  params: GridParams;
  parity: ParityTier;
}

// One fixed seed across every randomized algorithm, so the whole matrix is
// reproducible and shareable.
const RANDOM_STATE = 42;

// Target cluster count per dataset. The partitioning columns (K-Means, Spectral,
// Agglomerative, SOM) are forced to this k even on no-structure, where any split
// is arbitrary by construction.
const N_CLUSTERS: Record<GridDatasetId, number> = {
  moons: 2,
  circles: 2,
  blobs: 3,
  aniso: 3,
  none: 3,
};

function is_two_shape(id: GridDatasetId): boolean {
  return id === "moons" || id === "circles";
}

function kmeans_params(id: GridDatasetId): GridParams {
  return {
    algorithm_id: "kmeans",
    n_clusters: N_CLUSTERS[id],
    n_init: 10,
    random_state: RANDOM_STATE,
  };
}

// Connectivity affinity (nearest_neighbors) is what lets Spectral follow the
// curved manifolds of moons/circles and the stretched aniso clusters; RBF is the
// right kernel for the compact blobs and the structureless uniform field.
function spectral_params(id: GridDatasetId): GridParams {
  if (id === "moons" || id === "circles" || id === "aniso") {
    return {
      algorithm_id: "spectral",
      n_clusters: N_CLUSTERS[id],
      affinity: "nearest_neighbors",
      n_neighbors: 10,
      n_init: 10,
      random_state: RANDOM_STATE,
    };
  }
  return {
    algorithm_id: "spectral",
    n_clusters: N_CLUSTERS[id],
    affinity: "rbf",
    gamma: 1.0,
    n_init: 10,
    random_state: RANDOM_STATE,
  };
}

// Single linkage traces the thin moon/circle manifolds where Ward — biased toward
// compact balls — collapses them; Ward is the natural choice for the blob rows.
function agglomerative_params(id: GridDatasetId): GridParams {
  return {
    algorithm_id: "agglomerative",
    n_clusters: N_CLUSTERS[id],
    linkage: is_two_shape(id) ? "single" : "ward",
  };
}

// min_cluster_size scales with the local group size: small for the thin two-shape
// manifolds, mid for the blobs, and large for uniform no-structure — where a high
// floor is what lets HDBSCAN correctly report all-noise instead of inventing
// spurious density bumps.
function hdbscan_params(id: GridDatasetId): GridParams {
  if (is_two_shape(id)) {
    return { algorithm_id: "hdbscan", min_cluster_size: 10, min_samples: 5 };
  }
  if (id === "none") {
    return { algorithm_id: "hdbscan", min_cluster_size: 40, min_samples: 20 };
  }
  return { algorithm_id: "hdbscan", min_cluster_size: 15, min_samples: 5 };
}

function som_params(id: GridDatasetId): GridParams {
  const two = is_two_shape(id);
  return {
    algorithm_id: "som",
    grid_width: two ? 8 : 6,
    grid_height: two ? 8 : 6,
    num_epochs: 50,
    random_state: RANDOM_STATE,
    n_clusters: N_CLUSTERS[id],
    cluster_linkage: two ? "average" : "ward",
  };
}

function params_for(
  algorithm_id: GridAlgorithmId,
  dataset_id: GridDatasetId,
): GridParams {
  switch (algorithm_id) {
    case "kmeans":
      return kmeans_params(dataset_id);
    case "spectral":
      return spectral_params(dataset_id);
    case "agglomerative":
      return agglomerative_params(dataset_id);
    case "hdbscan":
      return hdbscan_params(dataset_id);
    case "som":
      return som_params(dataset_id);
  }
}

// The parity tier is data, not inference: SOM is always reference-free; uniform
// no-structure has no truth for any sklearn algorithm; Spectral and HDBSCAN drift
// at the float32 boundary on the shapes the task flags; everything else matches.
function parity_for(
  algorithm_id: GridAlgorithmId,
  dataset_id: GridDatasetId,
): ParityTier {
  if (algorithm_id === "som") return "no-reference";
  if (dataset_id === "none") return "no-truth";
  if (algorithm_id === "hdbscan") return "drifts";
  if (
    algorithm_id === "spectral" &&
    (dataset_id === "moons" ||
      dataset_id === "circles" ||
      dataset_id === "aniso")
  ) {
    return "drifts";
  }
  return "matches";
}

export function cell_id_of(
  dataset_id: GridDatasetId,
  algorithm_id: GridAlgorithmId,
): string {
  return `${dataset_id}:${algorithm_id}`;
}

// The 25 cells in render order (row-major: each dataset row, every algorithm).
export const GRID_CELLS: GridCell[] = GRID_DATASETS.flatMap((dataset) =>
  GRID_ALGORITHMS.map((algorithm) => ({
    cell_id: cell_id_of(dataset.id, algorithm.id),
    dataset_id: dataset.id,
    algorithm_id: algorithm.id,
    params: params_for(algorithm.id, dataset.id),
    parity: parity_for(algorithm.id, dataset.id),
  })),
);

export function count_parity(tier: ParityTier): number {
  return GRID_CELLS.filter((cell) => cell.parity === tier).length;
}
