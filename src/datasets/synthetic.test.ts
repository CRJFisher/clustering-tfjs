import { describe, it, expect } from "@jest/globals";
import * as tf from "../../test_support/tensorflow_helper";
import { make_blobs } from "./synthetic";

describe("make_blobs", () => {
  it("produces X and y with the requested shape", () => {
    const { X, y } = make_blobs({
      n_samples: 30,
      n_features: 2,
      centers: 3,
      random_state: 0,
    });
    expect(X.shape).toEqual([30, 2]);
    expect(y.length).toBe(30);
    X.dispose();
  });

  it("assigns one label per center across the full range", () => {
    const { X, y } = make_blobs({
      n_samples: 40,
      n_features: 4,
      centers: 4,
      random_state: 1,
    });
    expect(new Set(y)).toEqual(new Set([0, 1, 2, 3]));
    X.dispose();
  });

  it("spreads remainder samples onto the first clusters", () => {
    // 10 samples / 3 centers -> 4, 3, 3.
    const { X, y } = make_blobs({
      n_samples: 10,
      n_features: 2,
      centers: 3,
      random_state: 2,
    });
    const counts = [0, 0, 0];
    for (const label of y) counts[label]++;
    expect(counts).toEqual([4, 3, 3]);
    X.dispose();
  });

  it("is deterministic for a fixed random_state", () => {
    const a = make_blobs({ n_samples: 20, n_features: 3, centers: 2, random_state: 7 });
    const b = make_blobs({ n_samples: 20, n_features: 3, centers: 2, random_state: 7 });
    expect(Array.from(a.X.dataSync())).toEqual(Array.from(b.X.dataSync()));
    expect(a.y).toEqual(b.y);
    a.X.dispose();
    b.X.dispose();
  });

  it("places samples near explicit centers when cluster_std is tiny", () => {
    const centers = tf.tensor2d([
      [0, 0],
      [100, 100],
    ]);
    const { X, y } = make_blobs({
      n_samples: 20,
      n_features: 2,
      centers,
      cluster_std: 0.01,
      random_state: 3,
    });
    const data = X.arraySync() as number[][];
    data.forEach((point, i) => {
      const expected = y[i] === 0 ? [0, 0] : [100, 100];
      expect(point[0]).toBeCloseTo(expected[0], 1);
      expect(point[1]).toBeCloseTo(expected[1], 1);
    });
    X.dispose();
    centers.dispose();
  });

  it("does not dispose a caller-provided centers tensor", () => {
    const centers = tf.tensor2d([
      [1, 2],
      [3, 4],
    ]);
    const { X } = make_blobs({
      n_samples: 8,
      n_features: 2,
      centers,
      random_state: 4,
    });
    // Caller still owns `centers` and can read it after the call.
    expect(centers.isDisposed).toBe(false);
    expect(Array.from(centers.dataSync())).toEqual([1, 2, 3, 4]);
    X.dispose();
    centers.dispose();
  });

  it("leaks no tensors beyond the returned X", () => {
    const before = tf.memory().numTensors;
    const { X } = make_blobs({
      n_samples: 16,
      n_features: 2,
      centers: 2,
      random_state: 5,
    });
    expect(tf.memory().numTensors).toBe(before + 1);
    X.dispose();
  });
});
