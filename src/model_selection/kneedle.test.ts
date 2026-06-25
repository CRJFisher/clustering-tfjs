import { find_knee } from "./kneedle";

describe("find_knee", () => {
  it("detects knee in a typical WSS curve", () => {
    const k_values = [2, 3, 4, 5, 6, 7, 8];
    const wss_values = [100, 50, 30, 25, 23, 22, 21];

    const result = find_knee(k_values, wss_values, { direction: "concave" });

    expect(result.knee_x).toBe(4);
    expect(result.knee_index).toBe(2);
    expect(result.differences).toHaveLength(7);
  });

  it("returns null for perfectly linear data (all differences are zero)", () => {
    const k_values = [2, 3, 4, 5, 6];
    const wss_values = [50, 40, 30, 20, 10];

    const result = find_knee(k_values, wss_values, { direction: "concave" });

    expect(result.knee_x).toBeNull();
    expect(result.knee_index).toBeNull();
    // Confirm all differences are exactly zero
    for (const d of result.differences) {
      expect(d).toBeCloseTo(0, 10);
    }
  });

  it("returns null for fewer than 3 points", () => {
    const two = find_knee([2, 3], [100, 50]);
    expect(two.knee_x).toBeNull();
    expect(two.knee_index).toBeNull();
    expect(two.differences).toHaveLength(0);
    expect(find_knee([2], [100]).knee_x).toBeNull();
    expect(find_knee([], []).knee_x).toBeNull();
  });

  it("returns null and zero-filled differences for constant y values", () => {
    const result = find_knee([2, 3, 4, 5], [50, 50, 50, 50]);
    expect(result.knee_x).toBeNull();
    expect(result.differences).toEqual(new Array(4).fill(0));
  });

  it("returns null for constant x values", () => {
    const result = find_knee([3, 3, 3, 3], [100, 80, 60, 40]);
    expect(result.knee_x).toBeNull();
    expect(result.differences).toEqual(new Array(4).fill(0));
  });

  it("detects knee in a convex (increasing-then-flattening) curve", () => {
    const k_values = [2, 3, 4, 5, 6, 7];
    const wss_values = [10, 40, 60, 70, 73, 75];

    const result = find_knee(k_values, wss_values, { direction: "convex" });

    expect(result.knee_x).toBe(4);
    expect(result.knee_index).toBe(2);
  });

  it("high sensitivity suppresses a mild knee that low sensitivity finds", () => {
    // Mild WSS-like curve: max deviation ~0.15, threshold = S/(n-1)
    const k_values = [2, 3, 4, 5, 6, 7];
    const wss_values = [100, 95, 90, 85, 82, 80];

    // S=1 → threshold=0.2 > 0.15 → no knee
    const strict = find_knee(k_values, wss_values, {
      direction: "concave",
      sensitivity: 1,
    });
    expect(strict.knee_x).toBeNull();

    // S=0.5 → threshold=0.1 < 0.15 → knee detected
    const relaxed = find_knee(k_values, wss_values, {
      direction: "concave",
      sensitivity: 0.5,
    });
    expect(relaxed.knee_x).not.toBeNull();
  });

  it("differences array length matches input length even when no knee", () => {
    const result = find_knee([1, 2, 3], [10, 10, 10]);
    expect(result.differences).toHaveLength(3);
  });
});
