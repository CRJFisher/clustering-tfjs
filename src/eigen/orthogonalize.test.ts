import { describe, it, expect } from "@jest/globals";
import {
  reorthogonalize_vector,
  gram_schmidt_columns,
} from "./orthogonalize";

const dot = (a: ArrayLike<number>, b: ArrayLike<number>, n: number): number => {
  let s = 0;
  for (let i = 0; i < n; i++) s += a[i] * b[i];
  return s;
};

describe("reorthogonalize_vector", () => {
  it("projects out a single orthonormal basis vector", () => {
    const w = [2, 3, 4];
    const basis = [[1, 0, 0]];
    reorthogonalize_vector(w, basis, 3);
    expect(w).toEqual([0, 3, 4]);
  });

  it("leaves w orthogonal to every basis vector", () => {
    const w = [2, 3, 4];
    const basis = [
      [1, 0, 0],
      [0, 1, 0],
    ];
    reorthogonalize_vector(w, basis, 3);
    for (const q of basis) {
      expect(dot(w, q, 3)).toBeCloseTo(0, 12);
    }
    expect(w).toEqual([0, 0, 4]);
  });

  it("is a no-op when w is already orthogonal to the basis", () => {
    const w = [0, 0, 5];
    const basis = [
      [1, 0, 0],
      [0, 1, 0],
    ];
    reorthogonalize_vector(w, basis, 3);
    expect(w).toEqual([0, 0, 5]);
  });

  it("does nothing for an empty basis", () => {
    const w = [1, 2, 3];
    reorthogonalize_vector(w, [], 3);
    expect(w).toEqual([1, 2, 3]);
  });

  it("operates in place on Float64Array storage", () => {
    const w = new Float64Array([2, 3, 4]);
    const basis = [new Float64Array([1, 0, 0])];
    reorthogonalize_vector(w, basis, 3);
    expect(Array.from(w)).toEqual([0, 3, 4]);
  });
});

describe("gram_schmidt_columns", () => {
  it("leaves an already-orthonormal matrix unchanged", () => {
    const rows = [
      [1, 0],
      [0, 1],
    ];
    gram_schmidt_columns(rows, 2);
    expect(rows).toEqual([
      [1, 0],
      [0, 1],
    ]);
  });

  it("orthonormalizes the columns of a full-rank matrix", () => {
    // Columns are [1,1,0], [0,1,1], [0,0,1] (linearly independent).
    const rows = [
      [1, 0, 0],
      [1, 1, 0],
      [0, 1, 1],
    ];
    gram_schmidt_columns(rows, 3);

    const column = (j: number) => rows.map((r) => r[j]);
    for (let j = 0; j < 3; j++) {
      const c = column(j);
      expect(dot(c, c, 3)).toBeCloseTo(1, 12); // unit norm
    }
    for (let a = 0; a < 3; a++) {
      for (let b = a + 1; b < 3; b++) {
        expect(dot(column(a), column(b), 3)).toBeCloseTo(0, 12); // orthogonal
      }
    }
  });

  it("preserves the span of the original columns", () => {
    // The first orthonormal column is just the normalized first input column.
    const rows = [
      [3, 1],
      [4, 1],
    ];
    gram_schmidt_columns(rows, 2);
    expect(rows[0][0]).toBeCloseTo(3 / 5, 12);
    expect(rows[1][0]).toBeCloseTo(4 / 5, 12);
  });

  it("skips a degenerate (near-zero) column instead of dividing by zero", () => {
    const rows = [
      [0, 1],
      [0, 0],
    ];
    gram_schmidt_columns(rows, 2);
    // Column 0 is all zeros: left untouched, no NaN/Infinity introduced.
    expect(rows[0][0]).toBe(0);
    expect(rows[1][0]).toBe(0);
    expect(Number.isFinite(rows[0][1])).toBe(true);
    expect(Number.isFinite(rows[1][1])).toBe(true);
  });
});
