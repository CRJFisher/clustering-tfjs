import { GRID_CELLS } from "./grid_config";
import type { GridCell } from "./grid_config";
import {
  any_override_active,
  clamp_numeric,
  is_overridden,
  resolve_params,
} from "./grid_controls";

function cell(cell_id: string): GridCell {
  const found = GRID_CELLS.find((c) => c.cell_id === cell_id);
  if (!found) throw new Error(`No grid cell ${cell_id}`);
  return found;
}

describe("Auto state", () => {
  test("reproduces every cell's curated params untouched", () => {
    for (const grid_cell of GRID_CELLS) {
      expect(resolve_params(grid_cell, {})).toEqual(grid_cell.params);
      expect(is_overridden(grid_cell, {})).toBe(false);
    }
  });

  test("no override is active", () => {
    expect(any_override_active({})).toBe(false);
    expect(any_override_active({ n_clusters: undefined })).toBe(false);
  });
});

describe("n_clusters override", () => {
  test("retargets every partitioning column", () => {
    for (const algorithm of ["kmeans", "spectral", "agglomerative", "som"]) {
      const grid_cell = cell(`blobs:${algorithm}`);
      const params = resolve_params(grid_cell, { n_clusters: 5 });
      expect(params.algorithm_id).toBe(algorithm);
      if ("n_clusters" in params) expect(params.n_clusters).toBe(5);
      expect(is_overridden(grid_cell, { n_clusters: 5 })).toBe(true);
    }
  });

  test("leaves HDBSCAN untouched — it discovers its own count", () => {
    const grid_cell = cell("blobs:hdbscan");
    expect(resolve_params(grid_cell, { n_clusters: 5 })).toEqual(
      grid_cell.params,
    );
    expect(is_overridden(grid_cell, { n_clusters: 5 })).toBe(false);
  });

  test("setting the curated value back is not an override", () => {
    // blobs curates k=3; an explicit 3 resolves identically, so the badge stays.
    const grid_cell = cell("blobs:kmeans");
    expect(is_overridden(grid_cell, { n_clusters: 3 })).toBe(false);
  });
});

describe("Spectral affinity override", () => {
  test("forcing rbf onto a curated nearest_neighbors cell supplies a default gamma", () => {
    const grid_cell = cell("moons:spectral");
    expect(grid_cell.params).toMatchObject({ affinity: "nearest_neighbors" });
    const params = resolve_params(grid_cell, { spectral_affinity: "rbf" });
    expect(params).toMatchObject({ affinity: "rbf", gamma: 1.0 });
    expect("n_neighbors" in params && params.n_neighbors).toBeFalsy();
  });

  test("forcing nearest_neighbors onto a curated rbf cell supplies n_neighbors", () => {
    const grid_cell = cell("blobs:spectral");
    expect(grid_cell.params).toMatchObject({ affinity: "rbf" });
    const params = resolve_params(grid_cell, {
      spectral_affinity: "nearest_neighbors",
    });
    expect(params).toMatchObject({
      affinity: "nearest_neighbors",
      n_neighbors: 10,
    });
    expect("gamma" in params).toBe(false);
  });
});

describe("Spectral gamma override", () => {
  test("changes gamma on a cell whose effective affinity is rbf", () => {
    const grid_cell = cell("blobs:spectral");
    const params = resolve_params(grid_cell, { spectral_gamma: 4 });
    expect(params).toMatchObject({ affinity: "rbf", gamma: 4 });
    expect(is_overridden(grid_cell, { spectral_gamma: 4 })).toBe(true);
  });

  test("is inert on a nearest_neighbors cell — keeps its parity badge", () => {
    // gamma has no role under nearest_neighbors, so the cell is unchanged and must
    // not be marked exploratory.
    const grid_cell = cell("moons:spectral");
    expect(resolve_params(grid_cell, { spectral_gamma: 4 })).toEqual(
      grid_cell.params,
    );
    expect(is_overridden(grid_cell, { spectral_gamma: 4 })).toBe(false);
  });
});

describe("Agglomerative linkage override", () => {
  test("replaces the curated linkage across the column", () => {
    const grid_cell = cell("moons:agglomerative");
    expect(grid_cell.params).toMatchObject({ linkage: "single" });
    const params = resolve_params(grid_cell, {
      agglomerative_linkage: "ward",
    });
    expect(params).toMatchObject({ linkage: "ward" });
    expect(is_overridden(grid_cell, { agglomerative_linkage: "ward" })).toBe(
      true,
    );
  });
});

describe("HDBSCAN min_cluster_size override", () => {
  test("replaces the curated floor", () => {
    const grid_cell = cell("moons:hdbscan");
    const params = resolve_params(grid_cell, {
      hdbscan_min_cluster_size: 30,
    });
    expect(params).toMatchObject({ min_cluster_size: 30 });
    expect(is_overridden(grid_cell, { hdbscan_min_cluster_size: 30 })).toBe(
      true,
    );
  });
});

describe("SOM grid size override", () => {
  test("sets both lattice dimensions to a square map", () => {
    const grid_cell = cell("blobs:som");
    const params = resolve_params(grid_cell, { som_grid_size: 10 });
    expect(params).toMatchObject({ grid_width: 10, grid_height: 10 });
    expect(is_overridden(grid_cell, { som_grid_size: 10 })).toBe(true);
  });
});

describe("clamp_numeric", () => {
  test("clamps to each control's bounds", () => {
    expect(clamp_numeric("n_clusters", 99)).toBe(6);
    expect(clamp_numeric("n_clusters", 0)).toBe(2);
    expect(clamp_numeric("som_grid_size", 1)).toBe(4);
    expect(clamp_numeric("hdbscan_min_cluster_size", 1000)).toBe(60);
  });
});
