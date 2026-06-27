import {
  DEFAULT_BACKEND_ARG,
  generate_code,
  to_backend_arg,
} from "./generate_code";
import type { BackendArg } from "./generate_code";
import { GRID_CELLS } from "./grid_config";
import { resolve_params } from "./grid_controls";

const CLASS_BY_ALGORITHM: Record<string, string> = {
  kmeans: "KMeans",
  spectral: "SpectralClustering",
  agglomerative: "AgglomerativeClustering",
  hdbscan: "HDBSCAN",
  som: "SOM",
};

describe("every curated cell generates valid code", () => {
  test("each snippet imports the right class, inits a backend, and fits", () => {
    for (const cell of GRID_CELLS) {
      const code = generate_code({
        params: resolve_params(cell, {}),
        backend: DEFAULT_BACKEND_ARG,
      });
      const class_name = CLASS_BY_ALGORITHM[cell.algorithm_id];
      expect(code).toContain(
        `import { Clustering, ${class_name} } from "clustering-tfjs";`,
      );
      expect(code).toContain(`new ${class_name}(`);
      expect(code).toContain('await Clustering.init({ backend: "webgl" });');
    }
  });

  test("no snippet ever emits an `undefined` value", () => {
    for (const cell of GRID_CELLS) {
      const code = generate_code({
        params: resolve_params(cell, {}),
        backend: DEFAULT_BACKEND_ARG,
      });
      expect(code).not.toContain("undefined");
    }
  });
});

describe("per-algorithm shape", () => {
  function code_for(cell_id: string, backend: BackendArg = "webgl"): string {
    const cell = GRID_CELLS.find((c) => c.cell_id === cell_id);
    if (!cell) throw new Error(`No grid cell ${cell_id}`);
    return generate_code({ params: resolve_params(cell, {}), backend });
  }

  test("the four partitioning algorithms end in fit_predict", () => {
    for (const cell_id of [
      "blobs:kmeans",
      "blobs:spectral",
      "blobs:agglomerative",
      "blobs:hdbscan",
    ]) {
      expect(code_for(cell_id)).toContain("await model.fit_predict(X);");
    }
  });

  test("SOM uses the two-call fit-then-cluster path, not fit_predict", () => {
    const code = code_for("blobs:som");
    expect(code).toContain("await som.fit(X);");
    expect(code).toContain("await som.cluster(");
    expect(code).not.toContain("fit_predict");
  });
});

describe("spectral affinity branches", () => {
  test("an rbf cell emits gamma and no n_neighbors", () => {
    const cell = GRID_CELLS.find((c) => c.cell_id === "blobs:spectral");
    if (!cell) throw new Error("missing blobs:spectral");
    const code = generate_code({
      params: resolve_params(cell, { spectral_affinity: "rbf" }),
      backend: "webgl",
    });
    expect(code).toContain("gamma:");
    expect(code).not.toContain("n_neighbors:");
  });

  test("a nearest_neighbors cell emits n_neighbors and no gamma", () => {
    const cell = GRID_CELLS.find((c) => c.cell_id === "moons:spectral");
    if (!cell) throw new Error("missing moons:spectral");
    const code = generate_code({
      params: resolve_params(cell, { spectral_affinity: "nearest_neighbors" }),
      backend: "webgl",
    });
    expect(code).toContain("n_neighbors:");
    expect(code).not.toContain("gamma:");
  });
});

describe("overrides and backend flow into the snippet", () => {
  test("an n_clusters override appears in the constructor", () => {
    const cell = GRID_CELLS.find((c) => c.cell_id === "blobs:kmeans");
    if (!cell) throw new Error("missing blobs:kmeans");
    const code = generate_code({
      params: resolve_params(cell, { n_clusters: 5 }),
      backend: "webgl",
    });
    expect(code).toContain("n_clusters: 5");
  });

  test("the backend arg is reflected in the init line", () => {
    const cell = GRID_CELLS.find((c) => c.cell_id === "blobs:kmeans");
    if (!cell) throw new Error("missing blobs:kmeans");
    const code = generate_code({
      params: resolve_params(cell, {}),
      backend: "wasm",
    });
    expect(code).toContain('await Clustering.init({ backend: "wasm" });');
  });
});

describe("to_backend_arg", () => {
  test("passes through the three init-capable backends", () => {
    expect(to_backend_arg("webgl")).toBe("webgl");
    expect(to_backend_arg("wasm")).toBe("wasm");
    expect(to_backend_arg("cpu")).toBe("cpu");
  });

  test("rejects init-failure sentinels", () => {
    expect(to_backend_arg("none")).toBeUndefined();
    expect(to_backend_arg("timed out")).toBeUndefined();
    expect(to_backend_arg("worker error: boom")).toBeUndefined();
  });
});
