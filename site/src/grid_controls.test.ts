import { GRID_CELLS } from "./grid_config";
import type { GridCell } from "./grid_config";
import {
  any_override_active,
  clamp_numeric,
  is_overridden,
  params_equal,
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

describe("Spectral affinity + gamma together", () => {
  test("an explicit gamma wins over the default when switching to rbf", () => {
    const grid_cell = cell("moons:spectral");
    const params = resolve_params(grid_cell, {
      spectral_affinity: "rbf",
      spectral_gamma: 3,
    });
    expect(params).toMatchObject({ affinity: "rbf", gamma: 3 });
  });

  test("switching a curated rbf cell to rbf is a no-op — keeps its badge", () => {
    // blobs Spectral is curated rbf; redundantly selecting rbf must resolve to the
    // curated params (curated gamma preserved) and stay non-overridden.
    const grid_cell = cell("blobs:spectral");
    expect(resolve_params(grid_cell, { spectral_affinity: "rbf" })).toEqual(
      grid_cell.params,
    );
    expect(is_overridden(grid_cell, { spectral_affinity: "rbf" })).toBe(false);
  });
});

describe("setting a control back to its curated value is not an override", () => {
  test("HDBSCAN min_cluster_size at the curated floor", () => {
    // moons HDBSCAN curates min_cluster_size=10.
    const grid_cell = cell("moons:hdbscan");
    expect(is_overridden(grid_cell, { hdbscan_min_cluster_size: 10 })).toBe(
      false,
    );
  });

  test("SOM grid size at the curated lattice", () => {
    // blobs SOM curates a 6×6 lattice.
    const grid_cell = cell("blobs:som");
    expect(is_overridden(grid_cell, { som_grid_size: 6 })).toBe(false);
  });
});

describe("params_equal", () => {
  test("every cell's params equal themselves", () => {
    for (const grid_cell of GRID_CELLS) {
      expect(params_equal(grid_cell.params, grid_cell.params)).toBe(true);
    }
  });

  test("detects a single changed field", () => {
    const grid_cell = cell("blobs:som");
    const bigger = resolve_params(grid_cell, { som_grid_size: 10 });
    expect(params_equal(grid_cell.params, bigger)).toBe(false);
  });

  test("a switched Spectral affinity is unequal even at the same k", () => {
    const grid_cell = cell("moons:spectral");
    const as_rbf = resolve_params(grid_cell, { spectral_affinity: "rbf" });
    expect(params_equal(grid_cell.params, as_rbf)).toBe(false);
  });
});

describe("any_override_active", () => {
  test("is true once any field is set, ignoring undefined fields", () => {
    expect(any_override_active({ n_clusters: 4, spectral_gamma: undefined })).toBe(
      true,
    );
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
