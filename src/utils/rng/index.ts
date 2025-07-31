import { MT19937 } from './mt19937';

/** Lightweight interface used throughout the codebase for deterministic RNG. */
export interface RandomStream {
  /** Float in [0, 1). */
  rand(): number;
  /** Integer in [0, max). */
  randInt(max: number): number;
}

export function make_random_stream(seed?: number): RandomStream {
  if (seed === undefined) {
    return {
      rand: Math.random,
      randInt: (max: number) => Math.floor(Math.random() * max),
    };
  }

  const engine = new MT19937(seed >>> 0);
  return {
    rand: () => engine.nextFloat(),
    randInt: (max: number) => engine.nextInt(max),
  };
}
