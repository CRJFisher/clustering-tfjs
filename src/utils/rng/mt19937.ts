/**
 * Minimal TypeScript implementation of the original MT19937 32-bit variant
 * used by NumPy's legacy `RandomState` (and therefore by scikit-learn).
 *
 * This port only exposes the functionality required by the k-means++ seeding
 * routine:
 *   • Generation of 32-bit unsigned integers (\[0, 2**32))
 *   • High-precision uniform floats in the half-open interval \[0, 1)
 *
 * The algorithm closely follows the reference implementation described in
 * Matsumoto & Nishimura (1998) and the public domain C code.
 */

export class MT19937 {
  private static readonly N = 624;
  private static readonly M = 397;
  private static readonly MATRIX_A = 0x9908b0df;
  private static readonly UPPER_MASK = 0x80000000;
  private static readonly LOWER_MASK = 0x7fffffff;

  /** State vector – 624 32-bit unsigned ints. */
  private mt: Uint32Array = new Uint32Array(MT19937.N);

  /** Current index within the state vector. */
  private index = MT19937.N;

  constructor(seed: number) {
    this.init(seed >>> 0); // ensure unsigned 32-bit
  }

  /* --------------------------------------------------------------------- */
  /*                            Public helpers                               */
  /* --------------------------------------------------------------------- */

  /** Returns next 32-bit unsigned int in \[0, 2**32). */
  public nextUint32(): number {
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
   * Returns a 53-bit precision float in the interval \[0, 1) identical to
   * NumPy's `random_sample` implementation.
   */
  public nextFloat(): number {
    const a = this.nextUint32() >>> 5; // Upper 27 bits
    const b = this.nextUint32() >>> 6; // Upper 26 bits
    return (a * 67108864 + b) * 1.1102230246251565e-16; // 1 / 2**53
  }

  /** Uniform integer in \[0, max). Mirrors NumPy's rejection sampling to
   *  eliminate modulo bias so that sequences match exactly.
   */
  public nextInt(max: number): number {
    if (!Number.isInteger(max) || max <= 0 || max > 0xffffffff) {
      throw new Error('max must be a 32-bit positive integer');
    }

    const bound = max >>> 0;
    const threshold = (0x100000000 - bound) % bound; // 2**32 == 0x100000000

    // eslint-disable-next-line no-constant-condition
    while (true) {
      const r = this.nextUint32();
      if (r >= threshold) {
        return r % bound;
      }
    }
  }

  /* --------------------------------------------------------------------- */
  /*                               Internals                                */
  /* --------------------------------------------------------------------- */

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
