# Eigensolver Requirements for Spectral Clustering

## Matrix Characteristics

### Typical Matrix Sizes

Based on our test fixtures and expected use cases:

- **Small datasets**: 50x50 to 300x300 (current test fixtures)
- **Medium datasets**: 300x300 to 1000x1000 (typical clustering tasks)
- **Large datasets**: 1000x1000 to 5000x5000 (less common, performance sensitive)

Our failing test cases are relatively small:

- circles_n2_rbf: 100x100
- moons_n2_rbf: 100x100
- circles_n3_rbf: 150x150
- moons_n3_rbf: 150x150
- blobs_n3_rbf: 150x150

### Matrix Properties

**Yes, always symmetric**. We're computing eigenvalues of:

- Normalized Laplacian: L = I - D^(-1/2) _ A _ D^(-1/2)
- Where A is the affinity matrix (symmetric)
- The Laplacian is guaranteed symmetric by construction

### Eigenvalue Requirements

**We need only the smallest k eigenvalues/eigenvectors**, where k = number of clusters:

- Typically k = 2-10 (rarely more than 20)
- For spectral clustering, we specifically need the k smallest eigenvalues
- We do NOT need all eigenvalues - this is important for algorithm choice
- Currently using `smallest_eigenvectors_with_values()` to get just what we need

## Current Dependencies

### Already Using

- **TensorFlow.js** (@tensorflow/tfjs-node) - Primary numerical computation library
- We use tf tensors throughout for matrix operations
- Would prefer to stay within the TensorFlow.js ecosystem if possible

### Constraints

- **Pure JavaScript/TypeScript strongly preferred** for ease of distribution
- Current package is meant to be npm-installable without native dependencies
- Package size is a consideration but not critical (TensorFlow.js is already large)

## WebAssembly Considerations

**WebAssembly is definitely on the table** if it meets our needs:

### Pros of WASM approach:

- Platform independent (works in Node.js and browsers)
- No native compilation needed for end users
- Can achieve near-native performance
- Easier distribution via npm than native addons

### Cons to consider:

- Additional build complexity
- Potential startup overhead
- Need to manage memory crossing JS/WASM boundary
- Bundle size increase

### Our Position:

We would **strongly consider** a WebAssembly solution if it:

1. Provides accuracy matching ARPACK/sklearn (our main goal)
2. Handles symmetric matrices efficiently
3. Has a good API for finding just the k smallest eigenvalues
4. Integrates reasonably with TensorFlow.js tensors
5. Adds <10MB to bundle size

## Specific Requirements Summary

1. **Accuracy**: Must match sklearn's ARPACK to within 1e-4 (currently off by up to 0.0065)
2. **Matrix size**: Optimize for 100x100 to 1000x1000 symmetric matrices
3. **Eigenvalues**: Only need k smallest (k typically 2-10)
4. **Integration**: Must work with TensorFlow.js tensors
5. **Distribution**: Pure JS/TS or WASM preferred over native addons
6. **Performance**: Can be slower than current Jacobi if accuracy improves

## Recommendation Priority

1. **First choice**: Pure JS/TS implementation of Lanczos or Power Iteration
2. **Second choice**: WASM binding to proven C++ library (e.g., Spectra)
3. **Third choice**: Native addon if absolutely necessary for accuracy
4. **Last resort**: Accept current accuracy with better documentation

The key insight is we need high accuracy for finding just a few smallest eigenvalues of symmetric matrices - this is exactly what iterative methods like Lanczos are designed for.
