/// <reference types="jest" />
import { fit_log_domain, log_scale, nice_log_ticks } from "./chart_canvas";

describe("log_scale", () => {
  test("maps the domain endpoints to the range endpoints", () => {
    expect(log_scale(10, 10, 1000, 0, 200)).toBeCloseTo(0);
    expect(log_scale(1000, 10, 1000, 0, 200)).toBeCloseTo(200);
  });

  test("places a value at the geometric midpoint halfway along the range", () => {
    // 100 is the geometric mean of 10 and 1000, so it lands at the range centre.
    expect(log_scale(100, 10, 1000, 0, 200)).toBeCloseTo(100);
  });

  test("inverts the range for a descending pixel axis (top < bottom)", () => {
    // Bottom pixel for the smallest value, top pixel for the largest.
    expect(log_scale(10, 10, 1000, 200, 0)).toBeCloseTo(200);
    expect(log_scale(1000, 10, 1000, 200, 0)).toBeCloseTo(0);
  });
});

describe("nice_log_ticks", () => {
  test("emits 1-2-5 ticks across the bounding decades", () => {
    expect(nice_log_ticks(1, 100)).toEqual([1, 2, 5, 10, 20, 50, 100]);
  });

  test("includes only ticks within the range", () => {
    expect(nice_log_ticks(2, 50)).toEqual([2, 5, 10, 20, 50]);
  });

  test("returns nothing for a non-positive or inverted range", () => {
    expect(nice_log_ticks(0, 100)).toEqual([]);
    expect(nice_log_ticks(100, 1)).toEqual([]);
  });
});

describe("fit_log_domain", () => {
  test("snaps a data range out to its bounding decades", () => {
    expect(fit_log_domain([3, 40, 600])).toEqual([1, 1000]);
  });

  test("widens a single-decade range to a full decade so max > min", () => {
    const [lo, hi] = fit_log_domain([300, 700]);
    expect(lo).toBe(100);
    expect(hi).toBe(1000);
  });

  test("ignores non-positive values", () => {
    expect(fit_log_domain([0, -5, 12])).toEqual([10, 100]);
  });

  test("falls back to a default domain when there is no data", () => {
    expect(fit_log_domain([])).toEqual([1, 1000]);
  });
});
