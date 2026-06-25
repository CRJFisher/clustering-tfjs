/** Modifies `w` in place. */
export function reorthogonalize_vector<T extends { [i: number]: number; length: number }>(
  w: T,
  basis: T[],
  n: number,
): void {
  for (let b = 0; b < basis.length; b++) {
    const q = basis[b];
    let dot = 0;
    for (let i = 0; i < n; i++) dot += q[i] * w[i];
    for (let i = 0; i < n; i++) w[i] -= dot * q[i];
  }
}
