# Task 12 Subtask Reorganization Summary

## Changes Made

### Demoted Tasks (moved to drafts)
- **Former 12.5**: "Throw error instead of zero-padding embedding" 
  - Reason: We already handle this correctly; not causing test failures
- **Former 12.7**: "Complete randomState propagation throughout pipeline"
  - Reason: Jacobi solver is deterministic; consistent results show this isn't the issue

### Reordered/New Tasks (in priority order)

1. **Task 12.5** (NEW - HIGH PRIORITY): "Investigate RBF affinity calculation differences"
   - Focus on blobs_n2_rbf test with ARI=0.064
   - Compare RBF kernel computation with sklearn
   - Check gamma parameter interpretation and scaling

2. **Task 12.6** (NEW - HIGH PRIORITY): "Compare spectral embeddings with sklearn"
   - Export and compare embedding matrices directly
   - Look for normalization or processing differences
   - Investigate which eigenvectors are selected

3. **Task 12.7** (formerly 12.6 - MEDIUM PRIORITY): "Align k-means empty cluster handling with sklearn"
   - Keep this but lower priority
   - May help with edge cases but unlikely to fix systematic failures

## Rationale

Task 12.4.1's findings showed that eigensolver accuracy wasn't the bottleneck. Most failing tests have ARI > 0.7, suggesting we're close but have systematic differences in:
- RBF affinity calculation (especially given blobs_n2_rbf's very low ARI)
- Spectral embedding construction or usage
- Possibly k-means details (but less likely given consistent results)

This reorganization focuses effort on the most promising investigation paths based on empirical evidence.