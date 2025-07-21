# SpectralClustering Implementation Status

## Current State (2025-07-21)
- **7/12 fixture tests passing (58%)**
- All k-NN tests pass (6/6) ✅
- 5 RBF tests fail with ARI ~0.77-0.93 (need ≥0.95)
- 1 RBF test passes (blobs_n3_rbf)

## Root Cause Identified
Our Jacobi eigensolver produces slightly different eigenvectors than sklearn's ARPACK:
- Max difference: ~0.0065 per component
- These small differences cause k-means to produce different clusters
- Issue is deterministic - not related to random initialization

## Attempted Solutions
### ✅ Successful Fixes
1. **Component indicators for disconnected graphs** - All k-NN tests now pass
2. **Fixed normalized Laplacian computation** - Matches sklearn exactly
3. **Added diffusion map scaling** - Scale eigenvectors by sqrt(1 - eigenvalue)
4. **Fixed eigenvector selection** - Keep all eigenvectors including constant ones

### ❌ Unsuccessful Attempts
1. **Consensus clustering** (task 12.22)
   - Works for 2-cluster cases (perfect ARI)
   - Fails for 3-cluster cases (label switching issues)
2. **Increasing nInit** - No effect, results are deterministic
3. **Tighter Jacobi tolerances** - Already at 1e-14, can't improve further

## Remaining Options
1. **Alternative eigensolvers** (task 12.23) - Try power iteration or Lanczos
2. **Discretization** - Use sklearn's alternative label assignment method
3. **ARPACK bindings** - Native addon or WASM (complex)
4. **Accept current accuracy** - 87-93% ARI may be acceptable for pure JS

## Files Created
- `src/clustering/spectral_consensus.ts` - Consensus clustering implementation
- `src/utils/connected_components.ts` - Graph connectivity detection
- `src/utils/component_indicators.ts` - Component indicator vectors
- `src/utils/constant_eigenvector.ts` - Constant eigenvector utilities
- Various test and debug scripts

## Recommendation
Proceed with task 12.23 to investigate alternative eigensolvers. Power iteration or Lanczos methods may provide better numerical accuracy than Jacobi for finding smallest eigenvalues.