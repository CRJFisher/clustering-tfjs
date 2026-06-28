import type { GridAlgorithmId, GridParams } from "./grid_config";

// Generates the real, runnable ~5 lines a visitor would write to reproduce the
// clustering shown in a grid cell — the "every demo links to its source" payoff.
// The constructor calls mirror grid_worker.ts's `fit_labels` exactly, built from
// the cell's RESOLVED params (so live slider tweaks flow straight into the code)
// and the backend the demo actually initialized. Pure and DOM-free, so the per-
// algorithm output is unit-tested without a browser.

// The argument passed to Clustering.init — the live grid backend, which is one of
// the synchronous-readback-capable backends the grid worker chains through. This
// is the lowercase init arg, distinct from the uppercased display label.
export type BackendArg = "webgl" | "wasm" | "cpu";

// Used until the worker reports its live backend, so get_code() is valid the
// instant the panel mounts.
export const DEFAULT_BACKEND_ARG: BackendArg = "webgl";

export interface CodeRequest {
  params: GridParams;
  backend: BackendArg;
}

const CLASS_NAMES: Record<GridAlgorithmId, string> = {
  kmeans: "KMeans",
  spectral: "SpectralClustering",
  agglomerative: "AgglomerativeClustering",
  hdbscan: "HDBSCAN",
  som: "SOM",
};

type OptionValue = number | string;

// String options carry quotes (affinity, linkage); numeric options stay bare. The
// emitted text must parse as a real object literal — never an `undefined` value.
function format_options(entries: [string, OptionValue][]): string {
  return entries
    .map(([key, value]) =>
      typeof value === "string" ? `${key}: "${value}"` : `${key}: ${value}`,
    )
    .join(", ");
}

// The model lines for one algorithm, matching grid_worker.ts's constructors. SOM
// is the two-call exception (fit, then cluster down to k); every other algorithm
// shares the single `fit_predict` shape.
function model_lines(params: GridParams): string[] {
  switch (params.algorithm_id) {
    case "kmeans":
      return [
        `const model = new KMeans({ ${format_options([
          ["n_clusters", params.n_clusters],
          ["n_init", params.n_init],
          ["random_state", params.random_state],
        ])} });`,
        "const labels = await model.fit_predict(X);",
      ];
    case "spectral": {
      const entries: [string, OptionValue][] = [
        ["n_clusters", params.n_clusters],
        ["affinity", params.affinity],
      ];
      // Emit only the kernel parameter the chosen affinity actually uses, exactly
      // as the worker passes them — never a stray `gamma: undefined`.
      if (params.affinity === "rbf" && params.gamma !== undefined) {
        entries.push(["gamma", params.gamma]);
      }
      if (
        params.affinity === "nearest_neighbors" &&
        params.n_neighbors !== undefined
      ) {
        entries.push(["n_neighbors", params.n_neighbors]);
      }
      entries.push(
        ["n_init", params.n_init],
        ["random_state", params.random_state],
      );
      return [
        `const model = new SpectralClustering({ ${format_options(entries)} });`,
        "const labels = await model.fit_predict(X);",
      ];
    }
    case "agglomerative":
      return [
        `const model = new AgglomerativeClustering({ ${format_options([
          ["n_clusters", params.n_clusters],
          ["linkage", params.linkage],
        ])} });`,
        "const labels = await model.fit_predict(X);",
      ];
    case "hdbscan":
      return [
        `const model = new HDBSCAN({ ${format_options([
          ["min_cluster_size", params.min_cluster_size],
          ["min_samples", params.min_samples],
        ])} });`,
        "const labels = await model.fit_predict(X);",
      ];
    case "som":
      return [
        `const som = new SOM({ ${format_options([
          ["grid_width", params.grid_width],
          ["grid_height", params.grid_height],
          ["num_epochs", params.num_epochs],
          ["random_state", params.random_state],
        ])} });`,
        "await som.fit(X);",
        `const labels = await som.cluster(${params.n_clusters}, { linkage: "${params.cluster_linkage}" });`,
      ];
  }
}

export function generate_code(request: CodeRequest): string {
  const class_name = CLASS_NAMES[request.params.algorithm_id];
  return [
    `import { Clustering, ${class_name} } from "clustering-tfjs";`,
    "",
    `await Clustering.init({ backend: "${request.backend}" });`,
    "",
    "const X = [/* your data: number[n_samples][n_features] */];",
    ...model_lines(request.params),
  ].join("\n");
}

// Narrows the worker's reported backend label to an init arg. The grid chain only
// ever initializes one of these three (grid_worker BACKEND_CHAIN); init-failure
// sentinels ("none", "timed out", "worker error …") narrow to undefined so the
// panel keeps its default rather than emitting an un-runnable backend.
export function to_backend_arg(label: string): BackendArg | undefined {
  if (label === "webgl" || label === "wasm" || label === "cpu") return label;
  return undefined;
}
