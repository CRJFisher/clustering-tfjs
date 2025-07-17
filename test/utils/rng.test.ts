import { make_random_stream } from "../../src/utils/rng";

describe("RandomStream (MT19937)", () => {
  it("produces deterministic float & int sequences for same seed", () => {
    const rs1 = make_random_stream(123);
    const rs2 = make_random_stream(123);

    const floats1 = Array.from({ length: 5 }, () => rs1.rand());
    const floats2 = Array.from({ length: 5 }, () => rs2.rand());
    expect(floats1).toEqual(floats2);

    const ints1 = Array.from({ length: 5 }, () => rs1.randInt(1000));
    const ints2 = Array.from({ length: 5 }, () => rs2.randInt(1000));
    expect(ints1).toEqual(ints2);
  });

  it("different seeds yield different sequences", () => {
    const rs1 = make_random_stream(42);
    const rs2 = make_random_stream(43);

    const seq1 = Array.from({ length: 3 }, () => rs1.rand());
    const seq2 = Array.from({ length: 3 }, () => rs2.rand());

    expect(seq1).not.toEqual(seq2);
  });
});

