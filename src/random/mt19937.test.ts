import { MT19937 } from "./mt19937";

describe("MT19937", () => {
  // Golden values lock the deterministic output of the engine so that an
  // accidental change to the seeding, twist, or tempering steps is caught.
  it("produces the locked uint32 sequence for a fixed seed", () => {
    const m = new MT19937(42);
    const seq = Array.from({ length: 8 }, () => m.next_uint32());
    expect(seq).toEqual([
      698305938, 3044292490, 3355001886, 4224722126, 3517469812, 864223061,
      3864415249, 1160180227,
    ]);
  });

  it("produces the locked float sequence for a fixed seed", () => {
    const m = new MT19937(123);
    const seq = Array.from({ length: 8 }, () => m.next_float());
    expect(seq).toEqual([
      0.7838830193183882, 0.8498562332710481, 0.9340864610145109,
      0.364239921560934, 0.6604131585449591, 0.5189214992411124,
      0.7589819823965517, 0.5277359139055835,
    ]);
  });

  it("produces the locked bounded-int sequence for a fixed seed", () => {
    const m = new MT19937(7);
    const seq = Array.from({ length: 8 }, () => m.next_int(1000));
    expect(seq).toEqual([689, 198, 161, 436, 698, 307, 10, 694]);
  });

  it("emits unsigned 32-bit integers", () => {
    const m = new MT19937(99);
    for (let i = 0; i < 2000; i++) {
      const v = m.next_uint32();
      expect(Number.isInteger(v)).toBe(true);
      expect(v).toBeGreaterThanOrEqual(0);
      expect(v).toBeLessThanOrEqual(0xffffffff);
    }
  });

  it("emits floats in the half-open interval [0, 1)", () => {
    const m = new MT19937(99);
    for (let i = 0; i < 2000; i++) {
      const v = m.next_float();
      expect(v).toBeGreaterThanOrEqual(0);
      expect(v).toBeLessThan(1);
    }
  });

  it("emits bounded ints in [0, max) across the state-vector boundary", () => {
    // 700 draws exceed the 624-entry state vector, exercising twist() reseeding.
    const m = new MT19937(99);
    for (let i = 0; i < 700; i++) {
      const v = m.next_int(13);
      expect(Number.isInteger(v)).toBe(true);
      expect(v).toBeGreaterThanOrEqual(0);
      expect(v).toBeLessThan(13);
    }
  });

  it("gives identical streams for identical seeds", () => {
    const a = new MT19937(2024);
    const b = new MT19937(2024);
    const draws_a = Array.from({ length: 20 }, () => a.next_uint32());
    const draws_b = Array.from({ length: 20 }, () => b.next_uint32());
    expect(draws_a).toEqual(draws_b);
  });

  it("treats the seed as unsigned 32-bit", () => {
    // -1 >>> 0 === 0xffffffff, so a negative seed maps onto its unsigned value.
    const negative = new MT19937(-1);
    const unsigned = new MT19937(0xffffffff);
    const from_negative = Array.from({ length: 5 }, () => negative.next_uint32());
    const from_unsigned = Array.from({ length: 5 }, () => unsigned.next_uint32());
    expect(from_negative).toEqual(from_unsigned);
  });

  it("rejects an out-of-range bound for next_int", () => {
    const m = new MT19937(1);
    expect(() => m.next_int(0)).toThrow();
    expect(() => m.next_int(-5)).toThrow();
    expect(() => m.next_int(1.5)).toThrow();
    expect(() => m.next_int(0x1_0000_0000)).toThrow();
  });

  it("returns 0 for the degenerate bound of 1", () => {
    const m = new MT19937(1);
    for (let i = 0; i < 10; i++) {
      expect(m.next_int(1)).toBe(0);
    }
  });
});
