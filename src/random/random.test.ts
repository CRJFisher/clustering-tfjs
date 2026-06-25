import { make_random_stream } from "./";

describe("make_random_stream – unseeded", () => {
  it("returns values in [0, 1) and integer in [0, max) without seed", () => {
    const rs = make_random_stream();
    const v = rs.rand();
    expect(v).toBeGreaterThanOrEqual(0);
    expect(v).toBeLessThan(1);
    const n = rs.rand_int(100);
    expect(Number.isInteger(n)).toBe(true);
    expect(n).toBeGreaterThanOrEqual(0);
    expect(n).toBeLessThan(100);
  });
});

describe("RandomStream (MT19937)", () => {
  it("produces deterministic float & int sequences for same seed", () => {
    const rs1 = make_random_stream(123);
    const rs2 = make_random_stream(123);

    const floats1 = Array.from({ length: 5 }, () => rs1.rand());
    const floats2 = Array.from({ length: 5 }, () => rs2.rand());
    expect(floats1).toEqual(floats2);

    const ints1 = Array.from({ length: 5 }, () => rs1.rand_int(1000));
    const ints2 = Array.from({ length: 5 }, () => rs2.rand_int(1000));
    expect(ints1).toEqual(ints2);
  });

  it("different seeds yield different sequences", () => {
    const rs1 = make_random_stream(42);
    const rs2 = make_random_stream(43);

    const seq1 = Array.from({ length: 3 }, () => rs1.rand());
    const seq2 = Array.from({ length: 3 }, () => rs2.rand());

    expect(seq1).not.toEqual(seq2);
  });

  it("rand() values are in [0, 1)", () => {
    const rs = make_random_stream(7);
    for (let i = 0; i < 200; i++) {
      const v = rs.rand();
      expect(v).toBeGreaterThanOrEqual(0);
      expect(v).toBeLessThan(1);
    }
  });

  it("rand_int() values are non-negative integers strictly less than max", () => {
    const rs = make_random_stream(7);
    for (let i = 0; i < 200; i++) {
      const v = rs.rand_int(10);
      expect(Number.isInteger(v)).toBe(true);
      expect(v).toBeGreaterThanOrEqual(0);
      expect(v).toBeLessThan(10);
    }
  });

  it("rand_int(1) always returns 0", () => {
    const rs = make_random_stream(42);
    for (let i = 0; i < 50; i++) {
      expect(rs.rand_int(1)).toBe(0);
    }
  });

  it("negative seed is treated as its unsigned 32-bit equivalent (seed >>> 0)", () => {
    // -1 >>> 0 === 4294967295; both streams must produce identical sequences.
    const rs_neg = make_random_stream(-1);
    const rs_u32 = make_random_stream(4294967295);
    const seq_neg = Array.from({ length: 10 }, () => rs_neg.rand());
    const seq_u32 = Array.from({ length: 10 }, () => rs_u32.rand());
    expect(seq_neg).toEqual(seq_u32);
  });
});

