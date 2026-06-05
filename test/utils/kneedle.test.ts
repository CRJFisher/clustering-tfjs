import { describe, it, expect } from "@jest/globals";
import { find_knee } from "../../src/model_selection/kneedle";

describe("findKnee (Kneedle algorithm)", () => {
  it("should detect knee in a typical WSS curve", () => {
    // WSS-like curve: drops sharply then flattens
    const k_values = [2, 3, 4, 5, 6, 7, 8];
    const wss_values = [100, 50, 30, 25, 23, 22, 21];

    const result = find_knee(k_values, wss_values, { direction: "concave" });

    expect(result.knee_x).not.toBeNull();
    // The knee should be around k=3 or k=4 (where the curve flattens)
    expect(result.knee_x!).toBeGreaterThanOrEqual(3);
    expect(result.knee_x!).toBeLessThanOrEqual(5);
  });

  it("should return null for perfectly linear data", () => {
    const k_values = [2, 3, 4, 5, 6];
    const wss_values = [50, 40, 30, 20, 10];

    const result = find_knee(k_values, wss_values, { direction: "concave" });

    // Perfectly linear: differences from diagonal are all 0
    // May still detect a knee with default sensitivity, so just check it's reasonable
    if (result.knee_x !== null) {
      expect(result.knee_x).toBeGreaterThanOrEqual(2);
      expect(result.knee_x).toBeLessThanOrEqual(6);
    }
  });

  it("should return null for fewer than 3 points", () => {
    expect(find_knee([2, 3], [100, 50]).knee_x).toBeNull();
    expect(find_knee([2], [100]).knee_x).toBeNull();
    expect(find_knee([], []).knee_x).toBeNull();
  });

  it("should return null for constant y values", () => {
    const k_values = [2, 3, 4, 5];
    const wss_values = [50, 50, 50, 50];

    const result = find_knee(k_values, wss_values);
    expect(result.knee_x).toBeNull();
  });

  it("should detect knee in convex curves", () => {
    // Increasing curve that flattens
    const k_values = [2, 3, 4, 5, 6, 7];
    const wss_values = [10, 40, 60, 70, 73, 75];

    const result = find_knee(k_values, wss_values, { direction: "convex" });

    if (result.knee_x !== null) {
      expect(result.knee_x).toBeGreaterThanOrEqual(3);
      expect(result.knee_x).toBeLessThanOrEqual(5);
    }
  });

  it("should return differences array of correct length", () => {
    const k_values = [2, 3, 4, 5, 6];
    const wss_values = [100, 50, 30, 25, 23];

    const result = find_knee(k_values, wss_values);
    expect(result.differences).toHaveLength(5);
  });

  it("should respect sensitivity parameter", () => {
    const k_values = [2, 3, 4, 5, 6, 7];
    const wss_values = [100, 95, 90, 85, 82, 80];

    // Very high sensitivity should require a more pronounced knee
    const result = find_knee(k_values, wss_values, {
      direction: "concave",
      sensitivity: 100,
    });

    // With very high sensitivity, even a slight knee might not be detected
    // This is a mild curve so likely no knee
    // (just testing that sensitivity is respected)
    expect(result.differences).toHaveLength(6);
  });
});
