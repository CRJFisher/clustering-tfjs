/**
 * Minimal TypeScript implementation of the MT19937 32-bit generator described
 * in Matsumoto & Nishimura (1998) and the public domain C reference.
 *
 * Exposes only what the k-means++ seeding routine needs:
 *   • Generation of 32-bit unsigned integers (\[0, 2**32))
 *   • High-precision uniform floats in the half-open interval \[0, 1)
 *
 * Seeding uses the classic `init_genrand` recurrence. NumPy's legacy
 * `RandomState` instead expands an integer seed through `init_by_array`, so an
 * identical seed does not reproduce NumPy's stream here — only a deterministic
 * MT19937 stream of our own. k-means++ requires reproducible randomness, not
 * NumPy-identical randomness, so this is sufficient.
 */

export class MT19937 {
  private static readonly N = 624;
  private static readonly M = 397;
  private static readonly MATRIX_A = 0x9908b0df;
  private static readonly UPPER_MASK = 0x80000000;
  private static readonly LOWER_MASK = 0x7fffffff;

  private mt: Uint32Array = new Uint32Array(MT19937.N);
  private index = MT19937.N;

  constructor(seed: number) {
    this.init(seed >>> 0); // ensure unsigned 32-bit
  }

  public next_uint32(): number {
    if (this.index >= MT19937.N) {
      this.twist();
    }

    let y = this.mt[this.index++];

    // Tempering (same bit-shifts as NumPy's implementation)
    y ^= y >>> 11;
    y ^= (y << 7) & 0x9d2c5680;
    y ^= (y << 15) & 0xefc60000;
    y ^= y >>> 18;

    return y >>> 0; // ensure unsigned
  }

  /**
   * Returns a 53-bit precision float in the interval \[0, 1). The two-word
   * construction (27 high bits + 26 high bits scaled by 1/2**53) is the same
   * conversion formula NumPy's `random_sample` uses.
   */
  public next_float(): number {
    const a = this.next_uint32() >>> 5;
    const b = this.next_uint32() >>> 6;
    return (a * 67108864 + b) * 1.1102230246251565e-16; // 1 / 2**53
  }

  /** Uniform integer in \[0, max). Uses rejection sampling to discard the
   *  values that would otherwise fold unevenly under the modulo, eliminating
   *  modulo bias.
   */
  public next_int(max: number): number {
    if (!Number.isInteger(max) || max <= 0 || max > 0xffffffff) {
      throw new Error('max must be a 32-bit positive integer');
    }

    const bound = max >>> 0;
    const threshold = (0x100000000 - bound) % bound; // 2**32 == 0x100000000

    while (true) {
      const r = this.next_uint32();
      if (r >= threshold) {
        return r % bound;
      }
    }
  }

  private init(seed: number): void {
    this.mt[0] = seed >>> 0;
    for (let i = 1; i < MT19937.N; i++) {
      const prev = this.mt[i - 1] >>> 0;
      this.mt[i] =
        ((1812433253 * (prev ^ (prev >>> 30)) + i) & 0xffffffff) >>> 0;
    }
    this.index = MT19937.N;
  }

  private twist(): void {
    const { N, M, MATRIX_A, UPPER_MASK, LOWER_MASK } = MT19937;

    for (let i = 0; i < N; i++) {
      const x = (this.mt[i] & UPPER_MASK) | (this.mt[(i + 1) % N] & LOWER_MASK);
      let xa = x >>> 1;
      if (x % 2 !== 0) {
        xa ^= MATRIX_A;
      }
      this.mt[i] = this.mt[(i + M) % N] ^ xa;
    }

    this.index = 0;
  }
}
