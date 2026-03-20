---
id: task-38
title: Fix eigendecomposition numerical stability issues
status: To Do
assignee: []
created_date: '2026-03-20'
labels: []
dependencies: []
---

## Description

The eigendecomposition routines have critical numerical issues. (1) Division by zero in improved Jacobi when a_pp == a_qq and a_pq ~= 0, causing NaN propagation through entire eigenvector matrix. (2) PSD clamping zeroes valid small positive eigenvalues (threshold 1e-8 clamps positive values, not just negative), falsely signaling additional connected components. (3) Tridiagonal QR has variable shadowing bug where e[i] gets wrong value from outer scope g instead of inner loop g. (4) QR wilkinsonShift can produce NaN when delta==0 and b==0. (5) Loose tolerance (1e-2) in zero-eigenvalue detection causes eigenvalues up to 0.009 to be treated as component indicators.

## Acceptance Criteria

- [ ] Jacobi rotation handles a_pp == a_qq case without division by zero
- [ ] PSD clamping only zeroes negative eigenvalues not small positive ones
- [ ] Tridiagonal QR correctly updates e[i] from inner loop computation
- [ ] Wilkinson shift handles delta==0 b==0 edge case
- [ ] Zero-eigenvalue tolerance tightened to appropriate value
- [ ] Unit tests added for degenerate eigenvalue cases and edge conditions
- [ ] Eigenvector orthogonality validation added as post-condition check
