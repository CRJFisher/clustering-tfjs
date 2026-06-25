import { reorthogonalize_vector } from "./orthogonalize";

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
