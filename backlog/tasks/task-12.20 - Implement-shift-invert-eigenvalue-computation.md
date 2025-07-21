---
id: task-12.20
title: Implement shift-invert eigenvalue computation
status: To Do
assignee: []
created_date: '2025-07-21'
labels: []
dependencies: []
parent_task_id: task-12
---

## Description

Our spectral clustering fails on disconnected graphs because standard eigendecomposition produces eigenvectors with many unique values, while sklearn's shift-invert mode produces perfect component indicator eigenvectors (one unique value per component). This affects 7/12 failing fixture tests.

Shift-invert mode finds eigenvalues near a specified shift value (σ) by solving (A - σI)^(-1) instead of A directly. For spectral clustering, sklearn uses σ=1.0 on -L (negative Laplacian) to find eigenvalues near 0 of L.

## Acceptance Criteria

- [ ] Implement shift-invert eigenvalue solver for symmetric matrices
- [ ] Support finding k smallest eigenvalues of Laplacian
- [ ] Produce component indicator eigenvectors for disconnected graphs
- [ ] Pass all disconnected graph fixture tests (blobs_n2_knn, all RBF tests)
- [ ] Maintain performance for connected graphs
- [ ] Add comprehensive tests for the new solver

## Implementation Plan

### Option 1: Pure TypeScript Implementation (Recommended)

1. **Implement inverse iteration algorithm**:

   ```typescript
   // Pseudocode for shift-invert via inverse iteration
   function shiftInvertEigen(L: Tensor2D, k: number, sigma: number) {
     // 1. Form shifted matrix: M = L - σI
     const M = L.sub(tf.eye(n).mul(sigma));

     // 2. For each desired eigenvector:
     //    - Start with random vector
     //    - Repeatedly solve M * x_new = x_old
     //    - Normalize and check convergence

     // 3. Use Rayleigh quotient refinement for accuracy
   }
   ```

2. **Key challenges**:
   - Need efficient linear system solver for (L - σI)x = b
   - TensorFlow.js lacks built-in sparse matrix support
   - Must handle near-singular matrices carefully
   - Convergence can be slow without good initial guesses

3. **Implementation steps**:
   - Implement dense LU decomposition for linear solves
   - Add inverse iteration with deflation
   - Implement Rayleigh quotient iteration for refinement
   - Add convergence detection and error handling

### Option 2: WebAssembly Integration

1. **Use existing C++ library** (e.g., Eigen, Spectra):
   - Compile shift-invert solver to WASM
   - Create TypeScript bindings
   - Handle tensor conversion to/from WASM

2. **Advantages**:
   - Mature, optimized implementations
   - Better performance for large matrices
   - Sparse matrix support

3. **Disadvantages**:
   - Adds build complexity
   - Increases bundle size
   - May have platform compatibility issues

### Option 3: Approximate Solution (Pragmatic)

1. **Use power iteration on (I - L)**:

   ```typescript
   // For normalized Laplacian, largest eigenvalues of (I - L)
   // correspond to smallest eigenvalues of L
   function approximateShiftInvert(L: Tensor2D, k: number) {
     const I_minus_L = tf.eye(n).sub(L);
     // Use power iteration to find largest eigenvectors
     // These correspond to smallest eigenvectors of L
   }
   ```

2. **Component indicator enhancement**:
   - Detect number of components via near-zero eigenvalues
   - Use graph traversal to create explicit component indicators
   - Combine with standard eigenvectors for connected components

3. **Advantages**:
   - Simpler to implement
   - Works well for clearly disconnected graphs
   - No need for complex linear solvers

## Technical Details

### Why Shift-Invert Works

1. **Standard eigendecomposition**: Finds largest magnitude eigenvalues first
2. **Shift-invert**: Transforms spectrum so desired eigenvalues become largest
3. **For spectral clustering**:
   - Want smallest eigenvalues of L (near 0)
   - Shift-invert with σ=1 on -L makes these largest
   - Results in clean component indicators

### Component Indicator Pattern

For a graph with 3 components:

```txt
Standard eigenvectors:       Shift-invert eigenvectors:
[0.21, 0.19, 0.00, ...]     [0.33, 0.33, -0.17, ...]  # Constant per component
[0.00, 0.00, 0.24, ...]     [0.33, 0.33, -0.17, ...]
[0.00, 0.00, 0.19, ...]     [-0.17, -0.17, 0.46, ...]
```

### Performance Considerations

- Shift-invert requires solving linear systems (expensive)
- sklearn uses ARPACK's shift-invert mode (highly optimized)
- For small matrices (<1000 nodes), performance impact acceptable
- Consider caching decompositions for multiple eigenvalue finds

## Testing Strategy

1. **Unit tests**:
   - Compare with numpy/scipy results for small matrices
   - Test convergence on known problems
   - Verify component indicator property

2. **Integration tests**:
   - All failing fixture tests should pass
   - Performance benchmarks vs current implementation
   - Memory usage profiling

## References

- [ARPACK Users Guide](https://www.caam.rice.edu/software/ARPACK/UG/node45.html) - Shift-invert mode
- [SciPy eigsh documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.eigsh.html)
- [Numerical Linear Algebra](https://people.csail.mit.edu/steven/notes/numeric.pdf) - Inverse iteration

## Recommendation

Start with Option 3 (approximate solution) as it's most pragmatic:

1. Implement component detection via graph traversal
2. Create explicit component indicators when detected
3. Fall back to standard eigenvectors for connected graphs
4. This should fix all current failing tests

Later, if needed for more complex cases, implement Option 1 (pure TypeScript shift-invert).
