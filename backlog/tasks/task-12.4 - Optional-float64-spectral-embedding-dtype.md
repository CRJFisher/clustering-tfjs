# task-12.4 - Optional `dtype` parameter for float64 spectral embedding

## Description (the why)

TensorFlow.js defaults to `float32` tensors, which is generally sufficient for most deep-learning workloads but can introduce noticeable rounding error in linear-algebra heavy algorithms such as spectral clustering.  scikit-learn performs all computations in `float64`, so our `float32` pipeline accumulates small discrepancies that occasionally change k-means assignments on borderline points.  

Allowing the caller to opt-in to `float64` precision for the spectral embedding (while keeping the rest of the pipeline in `float32` for memory efficiency) will reduce these discrepancies and is the final step towards **bit-wise identical labels** in the reference tests.

## Acceptance Criteria (the what)

- [ ] `SpectralClusteringParams` gains optional `dtypeEmbedding?: "float32" | "float64"` (default `"float32"`).
- [ ] The Laplacian eigen solver and subsequent row-normalisation operate in the requested dtype, falling back to `float32` on WebGL back-ends that lack 64-bit support (with a clear warning).
- [ ] The downstream k-means step seamlessly consumes either precision without additional casts.
- [ ] Unit tests:
  - [ ] Switching to `float64` changes numeric results (embedding values) but **not** final cluster labels on reference datasets.
  - [ ] When backend does not support `float64`, implementation logs a single warning and proceeds in `float32`.
- [ ] Docs: update README and JSDoc to explain new parameter and backend caveats.

## Implementation Plan (the how)

1. Add `dtypeEmbedding` to params interface; update validation.
2. Inside spectral embedding routine, cast affinity / laplacian to desired dtype **after** validation but **before** eigendecomposition.
3. Use `tf.backend().floatPrecision()` or feature detection to decide if `float64` is supported; if not, emit `console.warn` once.
4. Ensure k-means receives `tf.tensor` with matching dtype or a plain `Float64Array` when `float64` is requested.
5. Update tests accordingly.

## Dependencies

Completes parity chain with tasks 12.1, 12.2 and 12.3.

## Implementation Notes (to fill after completion)

*TBD*

