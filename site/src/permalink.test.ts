import {
  PERMALINK_VERSION,
  decode_state,
  encode_state,
} from "./permalink";
import type { PermalinkState } from "./permalink";
import { N_MAX, N_MIN } from "./crossover";
import { NUMERIC_CONTROL_BOUNDS } from "./grid_controls";
import { GRID_ALGORITHMS, GRID_DATASETS } from "./grid_config";

function state(overrides: Partial<PermalinkState> = {}): PermalinkState {
  return {
    n: 2000,
    dataset_id: "blobs",
    algorithm_id: "spectral",
    overrides: {},
    ...overrides,
  };
}

describe("round-trip", () => {
  test("minimal state with all controls Auto", () => {
    const decoded = decode_state(encode_state(state()));
    expect(decoded.n).toBe(2000);
    expect(decoded.dataset_id).toBe("blobs");
    expect(decoded.algorithm_id).toBe("spectral");
    expect(decoded.overrides).toEqual({});
  });

  test("full state with every override set", () => {
    const full = state({
      n: 3200,
      dataset_id: "aniso",
      algorithm_id: "agglomerative",
      overrides: {
        n_clusters: 4,
        spectral_affinity: "rbf",
        spectral_gamma: 2.5,
        hdbscan_min_cluster_size: 25,
        agglomerative_linkage: "average",
        som_grid_size: 8,
      },
    });
    const decoded = decode_state(encode_state(full));
    expect(decoded.n).toBe(3200);
    expect(decoded.dataset_id).toBe("aniso");
    expect(decoded.algorithm_id).toBe("agglomerative");
    expect(decoded.overrides).toEqual(full.overrides);
  });

  test("nearest_neighbors affinity round-trips through its short code", () => {
    const decoded = decode_state(
      encode_state(state({ overrides: { spectral_affinity: "nearest_neighbors" } })),
    );
    expect(decoded.overrides.spectral_affinity).toBe("nearest_neighbors");
  });

  test("encoding is deterministic for equal states", () => {
    expect(encode_state(state())).toBe(encode_state(state()));
  });

  test("every dataset and algorithm id round-trips", () => {
    for (const dataset of GRID_DATASETS) {
      for (const algorithm of GRID_ALGORITHMS) {
        const decoded = decode_state(
          encode_state(
            state({ dataset_id: dataset.id, algorithm_id: algorithm.id }),
          ),
        );
        expect(decoded.dataset_id).toBe(dataset.id);
        expect(decoded.algorithm_id).toBe(algorithm.id);
      }
    }
  });

  test("every agglomerative linkage round-trips", () => {
    for (const linkage of ["ward", "complete", "average", "single"] as const) {
      const decoded = decode_state(
        encode_state(state({ overrides: { agglomerative_linkage: linkage } })),
      );
      expect(decoded.overrides.agglomerative_linkage).toBe(linkage);
    }
  });
});

describe("version gate", () => {
  test("unknown version drops everything despite valid fields", () => {
    const decoded = decode_state(
      `v=${PERMALINK_VERSION + 1}&n=1500&d=moons&a=kmeans`,
    );
    expect(decoded).toEqual({ overrides: {} });
  });

  test("missing version drops everything", () => {
    expect(decode_state("n=1500&d=moons")).toEqual({ overrides: {} });
  });

  test("non-numeric version drops everything", () => {
    expect(decode_state("v=abc&n=1500")).toEqual({ overrides: {} });
  });
});

describe("out-of-range numerics clamp to bounds", () => {
  test("n clamps to N_MIN and N_MAX", () => {
    expect(decode_state("v=1&n=50").n).toBe(N_MIN);
    expect(decode_state("v=1&n=99999").n).toBe(N_MAX);
  });

  test("n_clusters clamps to its bounds", () => {
    expect(decode_state("v=1&k=0").overrides.n_clusters).toBe(
      NUMERIC_CONTROL_BOUNDS.n_clusters.min,
    );
    expect(decode_state("v=1&k=99").overrides.n_clusters).toBe(
      NUMERIC_CONTROL_BOUNDS.n_clusters.max,
    );
  });

  test("hdbscan_min_cluster_size clamps to its bounds", () => {
    expect(decode_state("v=1&hm=1").overrides.hdbscan_min_cluster_size).toBe(
      NUMERIC_CONTROL_BOUNDS.hdbscan_min_cluster_size.min,
    );
    expect(decode_state("v=1&hm=500").overrides.hdbscan_min_cluster_size).toBe(
      NUMERIC_CONTROL_BOUNDS.hdbscan_min_cluster_size.max,
    );
  });

  test("som_grid_size clamps to its bounds", () => {
    expect(decode_state("v=1&som=1").overrides.som_grid_size).toBe(
      NUMERIC_CONTROL_BOUNDS.som_grid_size.min,
    );
    expect(decode_state("v=1&som=99").overrides.som_grid_size).toBe(
      NUMERIC_CONTROL_BOUNDS.som_grid_size.max,
    );
  });

  test("spectral_gamma clamps to its bounds", () => {
    expect(decode_state("v=1&sg=99").overrides.spectral_gamma).toBe(
      NUMERIC_CONTROL_BOUNDS.spectral_gamma.max,
    );
    expect(decode_state("v=1&sg=0").overrides.spectral_gamma).toBe(
      NUMERIC_CONTROL_BOUNDS.spectral_gamma.min,
    );
  });
});

describe("invalid enums are dropped, not applied", () => {
  test("bogus dataset id is omitted", () => {
    expect(decode_state("v=1&d=banana").dataset_id).toBeUndefined();
  });

  test("bogus algorithm id is omitted", () => {
    expect(decode_state("v=1&a=dbscan").algorithm_id).toBeUndefined();
  });

  test("bogus affinity code is omitted", () => {
    expect(decode_state("v=1&sa=cosine").overrides.spectral_affinity).toBeUndefined();
  });

  test("bogus linkage is omitted", () => {
    expect(decode_state("v=1&al=median").overrides.agglomerative_linkage).toBeUndefined();
  });

  test("per-field degradation is independent", () => {
    const decoded = decode_state("v=1&d=circles&a=dbscan");
    expect(decoded.dataset_id).toBe("circles");
    expect(decoded.algorithm_id).toBeUndefined();
  });
});

describe("non-numeric numeric fields are dropped, not clamped", () => {
  test("n=abc is omitted (keeps the slider default)", () => {
    expect(decode_state("v=1&n=abc").n).toBeUndefined();
  });

  test("k=xyz is omitted (stays Auto)", () => {
    expect(decode_state("v=1&k=xyz").overrides.n_clusters).toBeUndefined();
  });
});

describe("empty, partial, and leading-hash hashes", () => {
  test("empty hash decodes to defaults", () => {
    expect(decode_state("")).toEqual({ overrides: {} });
  });

  test("lone hash decodes to defaults", () => {
    expect(decode_state("#")).toEqual({ overrides: {} });
  });

  test("partial hash keeps only its valid fields", () => {
    const decoded = decode_state("v=1&d=circles");
    expect(decoded.dataset_id).toBe("circles");
    expect(decoded.n).toBeUndefined();
    expect(decoded.algorithm_id).toBeUndefined();
    expect(decoded.overrides).toEqual({});
  });

  test("a leading '#' is tolerated", () => {
    const without = decode_state("v=1&n=800&d=aniso&a=som");
    const with_hash = decode_state("#v=1&n=800&d=aniso&a=som");
    expect(with_hash).toEqual(without);
    expect(with_hash.n).toBe(800);
  });
});
