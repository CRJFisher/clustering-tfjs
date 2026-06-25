import * as tf from "../../test_support/tensorflow_helper";
import { build_contingency_table, to_label_array } from "./contingency";

describe("to_label_array", () => {
  it("returns a plain number array unchanged", () => {
    const labels = [0, 2, 1, 0];
    expect(to_label_array(labels)).toEqual([0, 2, 1, 0]);
  });

  it("converts a 1D tensor to a rounded integer array", () => {
    const labels = tf.tensor1d([0, 1, 2, 0.9999999, 1.0000001]);
    expect(to_label_array(labels)).toEqual([0, 1, 2, 1, 1]);
    labels.dispose();
  });
});

describe("build_contingency_table", () => {
  it("counts co-occurrences for perfectly aligned labels", () => {
    const result = build_contingency_table([0, 0, 1, 1], [0, 0, 1, 1]);
    expect(result.table).toEqual([
      [2, 0],
      [0, 2],
    ]);
    expect(result.row_sums).toEqual([2, 2]);
    expect(result.col_sums).toEqual([2, 2]);
    expect(result.n).toBe(4);
  });

  it("counts co-occurrences for a mixed assignment", () => {
    // true: A A B B, pred: x y x x  -> row A = {x:1,y:1}, row B = {x:2,y:0}
    const result = build_contingency_table([0, 0, 1, 1], [0, 1, 0, 0]);
    expect(result.table).toEqual([
      [1, 1],
      [2, 0],
    ]);
    expect(result.row_sums).toEqual([2, 2]);
    expect(result.col_sums).toEqual([3, 1]);
    expect(result.n).toBe(4);
  });

  it("maps arbitrary non-contiguous integer labels to dense indices", () => {
    // Labels 7 and 42 collapse onto rows 0 and 1 in first-seen order.
    const result = build_contingency_table([7, 42, 7, 42], [5, 5, 9, 9]);
    expect(result.table).toEqual([
      [1, 1],
      [1, 1],
    ]);
    expect(result.row_sums).toEqual([2, 2]);
    expect(result.col_sums).toEqual([2, 2]);
    expect(result.n).toBe(4);
  });

  it("preserves first-seen label ordering for table indices", () => {
    // First true label seen is 1, then 0 -> row 0 corresponds to label 1.
    const result = build_contingency_table([1, 0], [9, 8]);
    expect(result.table).toEqual([
      [1, 0],
      [0, 1],
    ]);
  });

  it("handles differing numbers of true and predicted classes", () => {
    const result = build_contingency_table([0, 0, 0], [0, 1, 2]);
    expect(result.table).toEqual([[1, 1, 1]]);
    expect(result.row_sums).toEqual([3]);
    expect(result.col_sums).toEqual([1, 1, 1]);
    expect(result.n).toBe(3);
  });

  it("returns empty structures for empty input", () => {
    const result = build_contingency_table([], []);
    expect(result.table).toEqual([]);
    expect(result.row_sums).toEqual([]);
    expect(result.col_sums).toEqual([]);
    expect(result.n).toBe(0);
  });

  it("row sums and column sums each total the sample count", () => {
    const result = build_contingency_table(
      [0, 1, 2, 0, 1, 2, 0],
      [1, 1, 0, 2, 0, 0, 1],
    );
    const total = (xs: number[]) => xs.reduce((a, b) => a + b, 0);
    expect(total(result.row_sums)).toBe(result.n);
    expect(total(result.col_sums)).toBe(result.n);
  });
});
