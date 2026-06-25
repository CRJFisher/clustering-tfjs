import { make_random_stream } from "./index";

describe("make_random_stream", () => {
  describe("seeded (deterministic)", () => {
    it("returns floats in [0, 1)", () => {
      const rng = make_random_stream(42);
      for (let i = 0; i < 200; i++) {
        const v = rng.rand();
        expect(v).toBeGreaterThanOrEqual(0);
        expect(v).toBeLessThan(1);
      }
    });

    it("returns integers in [0, max)", () => {
      const rng = make_random_stream(42);
      for (let i = 0; i < 200; i++) {
        const v = rng.rand_int(10);
        expect(Number.isInteger(v)).toBe(true);
        expect(v).toBeGreaterThanOrEqual(0);
        expect(v).toBeLessThan(10);
      }
    });

    it("produces identical streams for the same seed", () => {
      const a = make_random_stream(99);
      const b = make_random_stream(99);
      const floats_a = Array.from({ length: 10 }, () => a.rand());
      const floats_b = Array.from({ length: 10 }, () => b.rand());
      expect(floats_a).toEqual(floats_b);
    });

    it("produces different streams for different seeds", () => {
      const a = make_random_stream(1);
      const b = make_random_stream(2);
      const floats_a = Array.from({ length: 10 }, () => a.rand());
      const floats_b = Array.from({ length: 10 }, () => b.rand());
      expect(floats_a).not.toEqual(floats_b);
    });

    it("treats seed as unsigned 32-bit (negative seed maps to its unsigned value)", () => {
      // -1 >>> 0 === 0xffffffff, so new MT19937(-1) and new MT19937(0xffffffff) are identical.
      const neg = make_random_stream(-1);
      const uns = make_random_stream(0xffffffff);
      const from_neg = Array.from({ length: 5 }, () => neg.rand());
      const from_uns = Array.from({ length: 5 }, () => uns.rand());
      expect(from_neg).toEqual(from_uns);
    });

    it("forwards rand_int validation errors from MT19937", () => {
      const rng = make_random_stream(1);
      expect(() => rng.rand_int(0)).toThrow();
      expect(() => rng.rand_int(-1)).toThrow();
      expect(() => rng.rand_int(1.5)).toThrow();
    });
  });

  describe("unseeded (non-deterministic)", () => {
    it("returns floats in [0, 1)", () => {
      const rng = make_random_stream();
      for (let i = 0; i < 50; i++) {
        const v = rng.rand();
        expect(v).toBeGreaterThanOrEqual(0);
        expect(v).toBeLessThan(1);
      }
    });

    it("returns integers in [0, max)", () => {
      const rng = make_random_stream();
      for (let i = 0; i < 50; i++) {
        const v = rng.rand_int(7);
        expect(Number.isInteger(v)).toBe(true);
        expect(v).toBeGreaterThanOrEqual(0);
        expect(v).toBeLessThan(7);
      }
    });
  });
});
