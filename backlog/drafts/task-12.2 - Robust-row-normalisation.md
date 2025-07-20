---
id: task-12.2
title: Robust row normalisation with scipy-style clipping
status: To Do
assignee: []
created_date: '2025-07-18'
labels: [spectral]
dependencies: [task-12]
---

## Description (the why)

scikit-learn clips the per-row ℓ₂ norm used for normalising the spectral embedding:

```python
row_norm = np.maximum(np.linalg.norm(U, axis=1, keepdims=True), eps)
U = U / row_norm
```

We currently add `eps` after the division (`U / (row_norm + eps)`) which changes the vector *direction* when the norm is near zero and collapses several samples onto the same spot in low-variance datasets (e.g. the two tiny blobs robustness test).

## Acceptance Criteria (the what)

- [ ] Replace the division logic in `spectral.ts` with scikit-learn–style `max(row_norm, eps)` clipping.
- [ ] `eps` constant aligned with sklearn (`1e-10`).
- [ ] Existing unit tests remain green; the “two obvious blobs” robustness test turns green.
- [ ] Add micro test that a row of zeros stays zero after normalisation and does **not** introduce NaNs.

## Implementation Plan (the how)

1. Modify row-normalisation tidy block.
2. Re-run entire test-suite.

