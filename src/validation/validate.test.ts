import * as tf from "../../test_support/tensorflow_helper";
import {
  validate_labels_length,
  convert_validation_inputs,
  noise_filtered_indices,
} from "./validate";

describe("validate_labels_length", () => {
  it("passes when array lengths match", () => {
    expect(() =>
      validate_labels_length(
        [
          [1, 2],
          [3, 4],
        ],
        [0, 1],
      ),
    ).not.toThrow();
  });

  it("throws when array lengths differ", () => {
    expect(() =>
      validate_labels_length([[1, 2], [3, 4], [5, 6]], [0, 1]),
    ).toThrow(/Labels length \(2\) does not match data rows \(3\)/);
  });

  it("reads row counts from tensor inputs", () => {
    const X = tf.tensor2d([
      [1, 2],
      [3, 4],
    ]);
    const labels = tf.tensor1d([0, 1]);
    expect(() => validate_labels_length(X, labels)).not.toThrow();
    const bad = tf.tensor1d([0, 1, 2]);
    expect(() => validate_labels_length(X, bad)).toThrow();
    X.dispose();
    labels.dispose();
    bad.dispose();
  });

  it("accepts mixed tensor X with array labels", () => {
    const X = tf.tensor2d([[1, 2], [3, 4], [5, 6]]);
    expect(() => validate_labels_length(X, [0, 1, 2])).not.toThrow();
    expect(() => validate_labels_length(X, [0, 1])).toThrow();
    X.dispose();
  });
});

describe("convert_validation_inputs", () => {
  it("wraps an array matrix in a freshly-owned tensor", () => {
    const { data, label_array, owns_tensor } = convert_validation_inputs(
      [
        [1, 2],
        [3, 4],
      ],
      [0, 1],
    );
    expect(owns_tensor).toBe(true);
    expect(data.shape).toEqual([2, 2]);
    expect(label_array).toEqual([0, 1]);
    data.dispose();
  });

  it("passes a tensor matrix through without taking ownership", () => {
    const X = tf.tensor2d([
      [1, 2],
      [3, 4],
    ]);
    const { data, owns_tensor } = convert_validation_inputs(X, [0, 1]);
    expect(owns_tensor).toBe(false);
    expect(data).toBe(X); // same instance, caller still owns it
    X.dispose();
  });

  it("rounds tensor labels to integers", () => {
    const X = tf.tensor2d([
      [1, 2],
      [3, 4],
      [5, 6],
    ]);
    const labels = tf.tensor1d([0, 0.9999999, 2.0000001]);
    const { label_array } = convert_validation_inputs(X, labels);
    expect(label_array).toEqual([0, 1, 2]);
    X.dispose();
    labels.dispose();
  });

  it("passes an array label vector through unchanged", () => {
    const { label_array } = convert_validation_inputs(
      [
        [1, 2],
        [3, 4],
      ],
      [3, 7],
    );
    expect(label_array).toEqual([3, 7]);
  });

  it("array X + tensor labels: takes tensor ownership and rounds labels", () => {
    const labels = tf.tensor1d([0, 1.9999999]);
    const { data, label_array, owns_tensor } = convert_validation_inputs(
      [[1, 2], [3, 4]],
      labels,
    );
    expect(owns_tensor).toBe(true);
    expect(label_array).toEqual([0, 2]);
    data.dispose();
    labels.dispose();
  });
});

describe("noise_filtered_indices", () => {
  it("keeps every index when there is no noise", () => {
    expect(noise_filtered_indices([0, 1, 1, 2])).toEqual({
      keep: [0, 1, 2, 3],
      had_noise: false,
    });
  });

  it("drops -1 samples and flags that noise was present", () => {
    expect(noise_filtered_indices([0, -1, 1, -1, 2])).toEqual({
      keep: [0, 2, 4],
      had_noise: true,
    });
  });

  it("returns an empty keep set for all-noise input", () => {
    expect(noise_filtered_indices([-1, -1])).toEqual({
      keep: [],
      had_noise: true,
    });
  });

  it("handles empty input", () => {
    expect(noise_filtered_indices([])).toEqual({ keep: [], had_noise: false });
  });
});
