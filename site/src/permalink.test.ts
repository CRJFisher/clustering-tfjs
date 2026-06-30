import {
  PERMALINK_VERSION,
  decode_state,
  encode_state,
} from "./permalink";
import type { PermalinkState } from "./permalink";
import { GRID_ALGORITHMS, GRID_DATASETS } from "./grid_config";

function state(overrides: Partial<PermalinkState> = {}): PermalinkState {
  return {
    dataset_id: "blobs",
    algorithm_id: "spectral",
    ...overrides,
  };
}

describe("round-trip", () => {
  test("a selected cell round-trips through the hash", () => {
    const decoded = decode_state(encode_state(state()));
    expect(decoded.dataset_id).toBe("blobs");
    expect(decoded.algorithm_id).toBe("spectral");
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
});

describe("version gate", () => {
  test("unknown version drops everything despite valid fields", () => {
    const decoded = decode_state(`v=${PERMALINK_VERSION + 1}&d=moons&a=kmeans`);
    expect(decoded).toEqual({});
  });

  test("missing version drops everything", () => {
    expect(decode_state("d=moons&a=kmeans")).toEqual({});
  });

  test("non-numeric version drops everything", () => {
    expect(decode_state("v=abc&d=moons")).toEqual({});
  });
});

describe("invalid enums are dropped, not applied", () => {
  test("bogus dataset id is omitted", () => {
    expect(decode_state("v=1&d=banana").dataset_id).toBeUndefined();
  });

  test("bogus algorithm id is omitted", () => {
    expect(decode_state("v=1&a=dbscan").algorithm_id).toBeUndefined();
  });

  test("per-field degradation is independent", () => {
    const decoded = decode_state("v=1&d=circles&a=dbscan");
    expect(decoded.dataset_id).toBe("circles");
    expect(decoded.algorithm_id).toBeUndefined();
  });
});

describe("empty, partial, and leading-hash hashes", () => {
  test("empty hash decodes to defaults", () => {
    expect(decode_state("")).toEqual({});
  });

  test("lone hash decodes to defaults", () => {
    expect(decode_state("#")).toEqual({});
  });

  test("partial hash keeps only its valid fields", () => {
    const decoded = decode_state("v=1&d=circles");
    expect(decoded.dataset_id).toBe("circles");
    expect(decoded.algorithm_id).toBeUndefined();
  });

  test("a leading '#' is tolerated", () => {
    const without = decode_state("v=1&d=aniso&a=som");
    const with_hash = decode_state("#v=1&d=aniso&a=som");
    expect(with_hash).toEqual(without);
    expect(with_hash.dataset_id).toBe("aniso");
  });
});
