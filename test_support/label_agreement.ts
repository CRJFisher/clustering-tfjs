/**
 * Shared label-comparison helpers for clustering parity tests.
 *
 * Cluster ids are arbitrary, so reference comparisons must be invariant to
 * relabelling. Noise (`-1`) is identity, never permuted.
 */

/** Exact label equality up to a bijective cluster-id permutation (noise fixed). */
export function labels_equivalent_with_noise(
  a: number[],
  b: number[],
): boolean {
  if (a.length !== b.length) return false;
  const fwd = new Map<number, number>();
  const rev = new Map<number, number>();
  for (let i = 0; i < a.length; i++) {
    if ((a[i] === -1) !== (b[i] === -1)) return false;
    if (a[i] === -1) continue;
    if (fwd.has(a[i])) {
      if (fwd.get(a[i]) !== b[i]) return false;
    } else fwd.set(a[i], b[i]);
    if (rev.has(b[i])) {
      if (rev.get(b[i]) !== a[i]) return false;
    } else rev.set(b[i], a[i]);
  }
  return true;
}

/**
 * Cluster-assignment agreement in `[0, 1]` under the optimal greedy
 * cluster-id alignment, with noise (`-1`) treated as its own label. Used
 * where tie-ordering differences make exact equivalence too strict.
 */
export function alignment_agreement(mine: number[], reference: number[]): number {
  const pairs = new Map<string, number>();
  for (let i = 0; i < mine.length; i++) {
    const k = `${reference[i]}|${mine[i]}`;
    pairs.set(k, (pairs.get(k) ?? 0) + 1);
  }
  const map = new Map<number, number>();
  for (const s of new Set(reference)) {
    let best = -99;
    let bc = -1;
    for (const m of new Set(mine)) {
      const c = pairs.get(`${s}|${m}`) ?? 0;
      if (c > bc) {
        bc = c;
        best = m;
      }
    }
    map.set(s, best);
  }
  let ok = 0;
  for (let i = 0; i < mine.length; i++) {
    if (map.get(reference[i]) === mine[i]) ok++;
  }
  return ok / mine.length;
}
