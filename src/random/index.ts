import { MT19937 } from './mt19937';

/** Lightweight interface used throughout the codebase for deterministic RNG. */
export interface RandomStream {
  /** Float in [0, 1). */
  rand(): number;
  /** Integer in [0, max). */
  rand_int(max: number): number;
}

export function make_random_stream(seed?: number): RandomStream {
  if (seed === undefined) {
    return {
      rand: Math.random,
      rand_int: (max: number) => Math.floor(Math.random() * max),
    };
  }

  const engine = new MT19937(seed >>> 0);
  return {
    rand: () => engine.next_float(),
    rand_int: (max: number) => engine.next_int(max),
  };
}
