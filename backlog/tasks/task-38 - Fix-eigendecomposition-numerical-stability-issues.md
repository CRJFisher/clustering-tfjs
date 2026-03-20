---
id: task-38
title: Fix eigendecomposition numerical stability issues
status: Done
assignee: [claude]
created_date: '2026-03-20'
labels: []
dependencies: []
---

## Description

The eigendecomposition routines have critical numerical issues. (1) Division by zero in improved Jacobi when a_pp == a_qq and a_pq ~= 0, causing NaN propagation through entire eigenvector matrix. (2) PSD clamping zeroes valid small positive eigenvalues (threshold 1e-8 clamps positive values, not just negative), falsely signaling additional connected components. (3) Tridiagonal QR has variable shadowing bug where e[i] gets wrong value from outer scope g instead of inner loop g. (4) QR wilkinsonShift can produce NaN when delta==0 and b==0. (5) Loose tolerance (1e-2) in zero-eigenvalue detection causes eigenvalues up to 0.009 to be treated as component indicators.

## Acceptance Criteria

- [x] Jacobi rotation handles a_pp == a_qq case without division by zero
- [x] PSD clamping only zeroes negative eigenvalues not small positive ones
- [x] Tridiagonal QR correctly updates e[i] from inner loop computation
- [x] Wilkinson shift handles delta==0 b==0 edge case
- [x] Zero-eigenvalue tolerance tightened to appropriate value
- [x] Unit tests added for degenerate eigenvalue cases and edge conditions
- [x] Eigenvector orthogonality validation added as post-condition check

## Implementation Plan

1. Fix Jacobi div-by-zero in `eigen_improved.ts` by adding guard for near-zero `diff`
2. Fix PSD clamping in `eigen_improved.ts` to only clamp negative eigenvalues
3. Rewrite tridiagonal QR in `eigen_qr.ts` to match standard NR tqli algorithm
4. Add `b === 0` guard in Wilkinson shift + initialize `offDiag` from actual matrix norm
5. Tighten zero-eigenvalue tolerance from 1e-2/1e-5 to 1e-7
6. Add eigenvector orthogonality post-condition check to improved Jacobi
7. Add comprehensive test suite covering all edge cases

## Implementation Notes

### Fix 1: Jacobi div-by-zero (`eigen_improved.ts:89-108`)
Added three-branch rotation angle computation: when `|diff| <= EPSILON * max(|a_pp|, |a_qq|, 1)`, sets `t = sign(a_pq)` (standard Jacobi for equal diagonal elements, Golub & Van Loan). The existing small-angle and general branches are preserved for non-degenerate cases.

### Fix 2: PSD clamping (`eigen_improved.ts:174-190`)
Changed from `v < 1e-8 ? 0 : v` (which clamped valid small positives) to only clamping `v < 0` to zero. Added warning for large negative eigenvalues exceeding `tolerance * 100`. Also eliminated variable shadowing of outer `threshold`.

### Fix 3: Tridiagonal QR (`eigen_qr.ts:147-232`)
Rewrote the QL inner loop to match the standard Numerical Recipes tqli algorithm. The original code used a non-standard formulation (`c = (d[j] - shift) / r`) that was incompatible with the off-diagonal update formula. The corrected version uses `c = g / r` with `g` evolving through the loop as a `let` variable, eliminating the variable shadowing bug entirely. `e[i] = g` now correctly uses the last inner-loop value.

### Fix 4: Wilkinson shift NaN (`eigen_qr.ts:59`)
Added `if (b === 0) return c` guard — when the off-diagonal is zero, the 2x2 submatrix is already diagonal and the correct shift is `c`. Also initialized `offDiag` from actual matrix norm instead of `Infinity` to avoid unnecessary first iteration on already-diagonal matrices.

### Fix 5: Zero-eigenvalue tolerance
Changed `TOL` from `1e-2` to `1e-7` in `smallest_eigenvectors_with_values.ts` and from `1e-5` to `1e-7` in `eigen_improved.ts:laplacian_eigen_decomposition`. Eigenvalues as large as 0.009 were previously treated as connected component indicators.

### Fix 6: Eigenvector orthogonality validation (`eigen_improved.ts:210-222`)
Added post-condition check computing max |v_i · v_j| across all eigenvector pairs. Warns when orthogonality error exceeds 1e-6.

### Tests
Added `test/utils/eigen_stability.test.ts` with 18 tests covering:
- Equal diagonal elements (2x2, 4x4 Toeplitz, K3 Laplacian)
- PSD clamping (preserves small positives, clamps negatives, no-op without isPSD)
- Tridiagonal QR (3x3, identity, 2x2, reconstruction)
- Wilkinson shift NaN (diagonal, identity, repeated eigenvalues)
- Zero-eigenvalue tolerance (0.005 not zero, true zeros detected, 1e-6 boundary)
- Eigenvector orthogonality (Jacobi, tridiagonal QR)

### Modified files
- `src/utils/eigen_improved.ts` — Fixes 1, 2, 5, 6
- `src/utils/eigen_qr.ts` — Fixes 3, 4
- `src/utils/smallest_eigenvectors_with_values.ts` — Fix 5
- `test/utils/eigen_stability.test.ts` — New test file (18 tests)
