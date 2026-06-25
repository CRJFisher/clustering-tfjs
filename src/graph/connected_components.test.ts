import * as tf from "../../test_support/tensorflow_helper";

import {
  detect_connected_components,
  detect_sparse_connected_components,
  check_graph_connectivity,
} from "./connected_components";
import { sparse_matrix_from_row_maps } from "./sparse";

describe("detectConnectedComponents", () => {
  const tensors: tf.Tensor[] = [];

  function tracked<T extends tf.Tensor>(t: T): T {
    tensors.push(t);
    return t;
  }

  afterEach(() => {
    tensors.forEach((t) => t.dispose());
    tensors.length = 0;
  });

  it("finds a single component in a fully connected graph", () => {
    const affinity = tracked(
      tf.tensor2d([
        [1, 0.5, 0.3],
        [0.5, 1, 0.4],
        [0.3, 0.4, 1],
      ]),
    );
    const result = detect_connected_components(affinity);
    expect(result.num_components).toBe(1);
    expect(result.is_fully_connected).toBe(true);
    expect(result.component_labels).toEqual(new Int32Array([0, 0, 0]));
  });

  it("finds two components in a block-diagonal graph", () => {
    const affinity = tracked(
      tf.tensor2d([
        [1, 0.5, 0, 0],
        [0.5, 1, 0, 0],
        [0, 0, 1, 0.8],
        [0, 0, 0.8, 1],
      ]),
    );
    const result = detect_connected_components(affinity);
    expect(result.num_components).toBe(2);
    expect(result.is_fully_connected).toBe(false);
    // Nodes 0,1 share one label; nodes 2,3 share another
    expect(result.component_labels[0]).toBe(result.component_labels[1]);
    expect(result.component_labels[2]).toBe(result.component_labels[3]);
    expect(result.component_labels[0]).not.toBe(result.component_labels[2]);
  });

  it("finds three disconnected components", () => {
    const affinity = tracked(
      tf.tensor2d([
        [1, 0.5, 0, 0, 0],
        [0.5, 1, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 1, 0.7],
        [0, 0, 0, 0.7, 1],
      ]),
    );
    const result = detect_connected_components(affinity);
    expect(result.num_components).toBe(3);
    expect(result.is_fully_connected).toBe(false);
  });

  it("treats each isolated node as its own component", () => {
    // Identity matrix: no off-diagonal edges
    const affinity = tracked(
      tf.tensor2d([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
      ]),
    );
    const result = detect_connected_components(affinity);
    expect(result.num_components).toBe(3);
    expect(result.is_fully_connected).toBe(false);
    // Each node has a unique label
    const unique = new Set(result.component_labels);
    expect(unique.size).toBe(3);
  });

  it("handles a single-node graph", () => {
    const affinity = tracked(tf.tensor2d([[1]]));
    const result = detect_connected_components(affinity);
    expect(result.num_components).toBe(1);
    expect(result.is_fully_connected).toBe(true);
    expect(result.component_labels).toEqual(new Int32Array([0]));
  });

  it("respects custom tolerance (edges below tolerance ignored)", () => {
    // All off-diagonal values are 0.05 — below a tolerance of 0.1
    const affinity = tracked(
      tf.tensor2d([
        [1, 0.05, 0.05],
        [0.05, 1, 0.05],
        [0.05, 0.05, 1],
      ]),
    );
    const result = detect_connected_components(affinity, 0.1);
    expect(result.num_components).toBe(3);
    expect(result.is_fully_connected).toBe(false);
  });

  it("uses default tolerance of 1e-2 (boundary: 0.01 is NOT > 0.01)", () => {
    // Off-diagonal value exactly at 0.01 should NOT be treated as an edge
    // because the check is strictly greater than tolerance
    const affinity = tracked(
      tf.tensor2d([
        [1, 0.01],
        [0.01, 1],
      ]),
    );
    const result = detect_connected_components(affinity);
    expect(result.num_components).toBe(2);
    expect(result.is_fully_connected).toBe(false);
  });

  it("returns Int32Array for componentLabels", () => {
    const affinity = tracked(
      tf.tensor2d([
        [1, 0.5],
        [0.5, 1],
      ]),
    );
    const result = detect_connected_components(affinity);
    expect(result.component_labels).toBeInstanceOf(Int32Array);
  });

  it("assigns sequential labels starting from 0", () => {
    const affinity = tracked(
      tf.tensor2d([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0.5],
        [0, 0, 0.5, 1],
      ]),
    );
    const result = detect_connected_components(affinity);
    const labels = Array.from(result.component_labels);
    const unique = [...new Set(labels)].sort((a, b) => a - b);
    expect(unique[0]).toBe(0);
    for (let i = 1; i < unique.length; i++) {
      expect(unique[i]).toBe(unique[i - 1] + 1);
    }
  });
});

describe("checkGraphConnectivity", () => {
  const tensors: tf.Tensor[] = [];

  function tracked<T extends tf.Tensor>(t: T): T {
    tensors.push(t);
    return t;
  }

  afterEach(() => {
    tensors.forEach((t) => t.dispose());
    tensors.length = 0;
  });

  it("returns true for a connected graph without emitting a warning", () => {
    const warn_spy = jest.spyOn(console, "warn").mockImplementation(() => {});
    const affinity = tracked(
      tf.tensor2d([
        [1, 0.5],
        [0.5, 1],
      ]),
    );
    const result = check_graph_connectivity(affinity);
    expect(result).toBe(true);
    expect(warn_spy).not.toHaveBeenCalled();
    warn_spy.mockRestore();
  });

  it("returns false for a disconnected graph and emits a warning", () => {
    const warn_spy = jest.spyOn(console, "warn").mockImplementation(() => {});
    const affinity = tracked(
      tf.tensor2d([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
      ]),
    );
    const result = check_graph_connectivity(affinity);
    expect(result).toBe(false);
    expect(warn_spy).toHaveBeenCalled();
    warn_spy.mockRestore();
  });

  it("forwards custom tolerance to detection", () => {
    const warn_spy = jest.spyOn(console, "warn").mockImplementation(() => {});
    // Off-diagonal 0.05 — connected with default tolerance but not with 0.1
    const affinity = tracked(
      tf.tensor2d([
        [1, 0.05],
        [0.05, 1],
      ]),
    );
    const connected_default = check_graph_connectivity(affinity);
    expect(connected_default).toBe(true);

    const connected_strict = check_graph_connectivity(affinity, 0.1);
    expect(connected_strict).toBe(false);
    warn_spy.mockRestore();
  });
});

describe("detect_sparse_connected_components", () => {
  it("finds a single component in a connected sparse graph", () => {
    const A = sparse_matrix_from_row_maps([
      new Map([[1, 0.5]]),
      new Map([
        [0, 0.5],
        [2, 0.4],
      ]),
      new Map([[1, 0.4]]),
    ]);
    const result = detect_sparse_connected_components(A);
    expect(result.num_components).toBe(1);
    expect(result.is_fully_connected).toBe(true);
    expect(result.component_labels).toEqual(new Int32Array([0, 0, 0]));
  });

  it("finds two components in a block-diagonal sparse graph", () => {
    const A = sparse_matrix_from_row_maps([
      new Map([[1, 0.5]]),
      new Map([[0, 0.5]]),
      new Map([[3, 0.8]]),
      new Map([[2, 0.8]]),
    ]);
    const result = detect_sparse_connected_components(A);
    expect(result.num_components).toBe(2);
    expect(result.component_labels[0]).toBe(result.component_labels[1]);
    expect(result.component_labels[2]).toBe(result.component_labels[3]);
    expect(result.component_labels[0]).not.toBe(result.component_labels[2]);
  });

  it("treats isolated nodes as their own components", () => {
    const A = sparse_matrix_from_row_maps([new Map(), new Map(), new Map()]);
    const result = detect_sparse_connected_components(A);
    expect(result.num_components).toBe(3);
    expect(new Set(result.component_labels).size).toBe(3);
  });

  it("ignores edges at or below the tolerance", () => {
    const A = sparse_matrix_from_row_maps([
      new Map([[1, 0.05]]),
      new Map([[0, 0.05]]),
    ]);
    expect(detect_sparse_connected_components(A).num_components).toBe(1);
    expect(detect_sparse_connected_components(A, 0.1).num_components).toBe(2);
  });

  it("matches the dense detector on the same graph", () => {
    const dense = tf.tensor2d([
      [1, 0.5, 0, 0],
      [0.5, 1, 0, 0],
      [0, 0, 1, 0.7],
      [0, 0, 0.7, 1],
    ]);
    const sparse = sparse_matrix_from_row_maps([
      new Map([[1, 0.5]]),
      new Map([[0, 0.5]]),
      new Map([[3, 0.7]]),
      new Map([[2, 0.7]]),
    ]);
    const dense_result = detect_connected_components(dense);
    const sparse_result = detect_sparse_connected_components(sparse);
    expect(sparse_result.num_components).toBe(dense_result.num_components);
    expect(sparse_result.component_labels).toEqual(
      dense_result.component_labels,
    );
    dense.dispose();
  });

  it("throws on a non-square affinity", () => {
    const rect = sparse_matrix_from_row_maps([new Map([[0, 1]])], 4);
    expect(() => detect_sparse_connected_components(rect)).toThrow("square");
  });
});
