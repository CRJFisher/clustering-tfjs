import { describe, it, expect } from "@jest/globals";
import * as tf from "../../test_support/tensorflow_helper";
import {
  grid_to_index,
  index_to_grid,
  grid_distance,
  get_neighbors,
  create_grid_distance_matrix,
  get_grid_coordinates,
  find_bmu,
  find_bmu_batch,
  compute_bmu_distances,
  gaussian_neighborhood,
  bubble_neighborhood,
  mexican_hat_neighborhood,
  validate_neighborhood_params,
  linear_decay,
  exponential_decay,
  inverse_time_decay,
  create_decay_scheduler,
  decay_tensor,
  DecayTracker,
  adaptive_radius,
  adaptive_learning_rate,
} from "./som_neighborhood";

describe("grid coordinate conversions", () => {
  it("round-trips between flat index and grid coordinates", () => {
    expect(grid_to_index(2, 3, 5)).toBe(13);
    expect(index_to_grid(13, 5)).toEqual([2, 3]);
    expect(index_to_grid(grid_to_index(4, 1, 7), 7)).toEqual([4, 1]);
  });
});

describe("grid_distance", () => {
  it("uses plain Euclidean distance on a rectangular grid", () => {
    expect(grid_distance([0, 0], [3, 4], "rectangular")).toBeCloseTo(5, 12);
    expect(grid_distance([1, 1], [1, 1], "rectangular")).toBe(0);
  });

  it("treats adjacent hexagonal cells as unit distance", () => {
    // Same row and the offset diagonal neighbour are both distance 1.
    expect(grid_distance([0, 0], [0, 1], "hexagonal")).toBeCloseTo(1, 12);
    expect(grid_distance([0, 0], [1, 0], "hexagonal")).toBeCloseTo(1, 12);
  });
});

describe("get_neighbors", () => {
  it("returns the full 8-neighbourhood for an interior rectangular cell", () => {
    const n = get_neighbors(1, 1, 3, 3, "rectangular");
    expect(n.length).toBe(8);
    expect(n).not.toContainEqual([1, 1]);
  });

  it("clips rectangular neighbours at grid corners", () => {
    const n = get_neighbors(0, 0, 3, 3, "rectangular");
    expect(new Set(n.map((p) => p.join(",")))).toEqual(
      new Set(["0,1", "1,0", "1,1"]),
    );
  });

  it("uses parity-dependent offsets on a hexagonal grid", () => {
    const even = get_neighbors(0, 0, 3, 3, "hexagonal");
    expect(new Set(even.map((p) => p.join(",")))).toEqual(
      new Set(["0,1", "1,0"]),
    );
    const odd = get_neighbors(1, 1, 3, 3, "hexagonal");
    expect(odd.length).toBe(6);
  });
});

describe("create_grid_distance_matrix / get_grid_coordinates", () => {
  it("builds a symmetric zero-diagonal distance matrix", () => {
    const m = create_grid_distance_matrix(2, 2, "rectangular");
    const data = m.arraySync() as number[][];
    const total = 4;
    for (let i = 0; i < total; i++) {
      expect(data[i][i]).toBe(0);
      for (let j = 0; j < total; j++) {
        expect(data[i][j]).toBeCloseTo(data[j][i], 12);
      }
    }
    m.dispose();
  });

  it("returns one coordinate pair per neuron", () => {
    const c = get_grid_coordinates(2, 3, "rectangular");
    expect(c.shape).toEqual([6, 2]);
    expect((c.arraySync() as number[][])[0]).toEqual([0, 0]);
    c.dispose();
  });
});

describe("BMU search", () => {
  // Weights: neuron (row, col) holds vector [row*10, col*10].
  const make_weights = () =>
    tf.tensor3d([
      [
        [0, 0],
        [0, 10],
      ],
      [
        [10, 0],
        [10, 10],
      ],
    ]);

  it("find_bmu selects the closest neuron for a single sample", () => {
    const weights = make_weights();
    const sample = tf.tensor1d([9, 11]);
    const bmu = find_bmu(sample, weights);
    expect(Array.from(bmu.dataSync())).toEqual([1, 1]);
    bmu.dispose();
    sample.dispose();
    weights.dispose();
  });

  it("find_bmu_batch matches find_bmu per row", () => {
    const weights = make_weights();
    const samples = tf.tensor2d([
      [0, 1],
      [9, 1],
      [1, 9],
    ]);
    const batch = find_bmu_batch(samples, weights);
    expect(batch.arraySync()).toEqual([
      [0, 0],
      [1, 0],
      [0, 1],
    ]);
    batch.dispose();
    samples.dispose();
    weights.dispose();
  });

  it("compute_bmu_distances returns the sample-to-BMU distance", () => {
    const weights = make_weights();
    const samples = tf.tensor2d([
      [0, 0],
      [13, 10],
    ]);
    const bmus = tf.tensor2d([
      [0, 0],
      [1, 1],
    ]);
    const dist = compute_bmu_distances(samples, weights, bmus);
    const values = Array.from(dist.dataSync());
    expect(values[0]).toBeCloseTo(0, 6);
    expect(values[1]).toBeCloseTo(3, 6); // [13,10] vs neuron [10,10]
    dist.dispose();
    weights.dispose();
    samples.dispose();
    bmus.dispose();
  });
});

describe("neighborhood functions", () => {
  it("gaussian peaks at the BMU and decays with distance", () => {
    const d = tf.tensor1d([0, 1, 2]);
    const h = gaussian_neighborhood(d, 1) as tf.Tensor1D;
    const v = Array.from(h.dataSync());
    expect(v[0]).toBeCloseTo(1, 6);
    expect(v[0]).toBeGreaterThan(v[1]);
    expect(v[1]).toBeGreaterThan(v[2]);
    d.dispose();
    h.dispose();
  });

  it("bubble is a hard 0/1 cutoff at the radius", () => {
    const d = tf.tensor1d([0, 1, 2, 3]);
    const h = bubble_neighborhood(d, 2) as tf.Tensor1D;
    expect(Array.from(h.dataSync())).toEqual([1, 1, 1, 0]);
    d.dispose();
    h.dispose();
  });

  it("mexican hat is positive at the centre and negative in the surround", () => {
    const d = tf.tensor1d([0, 2]);
    const h = mexican_hat_neighborhood(d, 1) as tf.Tensor1D;
    const v = Array.from(h.dataSync());
    expect(v[0]).toBeCloseTo(2, 6); // amplitude * (1 - 0) * exp(0)
    expect(v[1]).toBeLessThan(0); // 1 - (2)^2 < 0
    d.dispose();
    h.dispose();
  });
});

describe("validate_neighborhood_params", () => {
  it("rejects a non-positive radius", () => {
    expect(() => validate_neighborhood_params(0, 5, 5)).toThrow();
    expect(() => validate_neighborhood_params(-1, 5, 5)).toThrow();
  });

  it("accepts a radius within the grid", () => {
    expect(() => validate_neighborhood_params(2, 5, 5)).not.toThrow();
  });
});

describe("decay functions", () => {
  it("linear_decay interpolates from initial to final", () => {
    expect(linear_decay(10, 0, 0, 11)).toBeCloseTo(10, 12);
    expect(linear_decay(10, 0, 5, 11)).toBeCloseTo(5, 12);
    expect(linear_decay(10, 0, 10, 11)).toBeCloseTo(0, 12);
    expect(linear_decay(10, 0, 3, 1)).toBe(10); // degenerate total_epochs
  });

  it("exponential_decay starts at initial and approaches final", () => {
    expect(exponential_decay(10, 1, 0, 100)).toBeCloseTo(10, 12);
    expect(exponential_decay(10, 1, 100, 100)).toBeLessThan(10);
    expect(exponential_decay(10, 1, 100, 100)).toBeGreaterThan(1);
  });

  it("inverse_time_decay starts at initial and is monotonically decreasing", () => {
    expect(inverse_time_decay(10, 1, 0, 100)).toBeCloseTo(10, 12);
    const a = inverse_time_decay(10, 1, 10, 100);
    const b = inverse_time_decay(10, 1, 20, 100);
    expect(a).toBeGreaterThan(b);
  });
});

describe("create_decay_scheduler", () => {
  it("returns a custom decay function unchanged", () => {
    const custom = (epoch: number) => epoch * 2;
    expect(create_decay_scheduler(custom, "linear", 10)).toBe(custom);
  });

  it("builds a linear schedule honouring the final value", () => {
    const fn = create_decay_scheduler(1, "linear", 11, 0);
    expect(fn(0, 11)).toBeCloseTo(1, 12);
    expect(fn(10, 11)).toBeCloseTo(0, 12);
  });

  it("throws on an unknown strategy", () => {
    expect(() =>
      // @ts-expect-error deliberately invalid strategy
      create_decay_scheduler(1, "quadratic", 10),
    ).toThrow();
  });
});

describe("decay_tensor", () => {
  it("matches the scalar linear formula at the endpoints", () => {
    const start = decay_tensor(10, 0, 0, 11, "linear");
    const end = decay_tensor(10, 0, 10, 11, "linear");
    expect(start.dataSync()[0]).toBeCloseTo(10, 6);
    expect(end.dataSync()[0]).toBeCloseTo(0, 6);
    start.dispose();
    end.dispose();
  });
});

describe("DecayTracker", () => {
  it("advances epochs and records history", () => {
    const tracker = new DecayTracker(10, "linear", 11, 0);
    const first = tracker.next(11);
    const second = tracker.next(11);
    expect(first).toBeCloseTo(10, 12);
    expect(second).toBeLessThan(first);
    expect(tracker.get_epoch()).toBe(2);
    expect(tracker.get_history().length).toBe(2);
  });

  it("current() does not advance the epoch", () => {
    const tracker = new DecayTracker(10, "linear", 11, 0);
    tracker.current(11);
    expect(tracker.get_epoch()).toBe(0);
  });

  it("reset() clears epoch and history", () => {
    const tracker = new DecayTracker(10, "linear", 11, 0);
    tracker.next(11);
    tracker.reset();
    expect(tracker.get_epoch()).toBe(0);
    expect(tracker.get_history()).toEqual([]);
  });
});

describe("adaptive helpers", () => {
  it("adaptive_radius starts near half the grid extent", () => {
    expect(adaptive_radius(10, 10, 0, 100)).toBeCloseTo(5, 12);
  });

  it("adaptive_learning_rate starts at the initial rate and decays", () => {
    expect(adaptive_learning_rate(0.5, 0, 100)).toBeCloseTo(0.5, 12);
    expect(adaptive_learning_rate(0.5, 100, 100)).toBeLessThan(0.5);
  });
});
