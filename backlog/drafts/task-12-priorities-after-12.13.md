# Task 12 Sub-task Priorities After Task 12.13 Findings

Date: 2025-07-21

## Executive Summary

After implementing diffusion map scaling in Task 12.13, we've fixed all disconnected component cases (blobs datasets) but still have 8/12 fixtures failing. This document re-evaluates the remaining sub-tasks based on our latest understanding.

## Current Status

- **4/12 fixtures passing** (all blobs datasets with ARI = 1.0)
- **8/12 fixtures failing** (circles and moons with ARI between 0.31 and 0.95)
- **Key finding**: sklearn uses diffusion map scaling, not eigenvector recovery
- **Main issue**: k-NN graphs create structural problems that prevent ideal embeddings

## Revised Priority Order

### Priority 1: Investigate k-NN Graph Structure Issues

**Task 12.19** - Test with sklearn's exact parameters and data

- **Why first**: We need to understand exactly what sklearn is doing differently
- **Expected outcome**: Identify the missing piece that allows sklearn to handle k-NN graphs better
- **Impact**: Could unlock the solution for all remaining tests

### Priority 2: Fix Known Scaling Issues

**Task 12.17** - Fix RBF gamma scaling for fixture data

- **Why second**: This is a known issue with a clear fix
- **Expected outcome**: Improve RBF test results (potentially fix circles_n2_rbf, moons_n2_rbf)
- **Impact**: Could fix 2-3 more tests

### Priority 3: Handle Edge Cases Better

**Merge Tasks 12.15 & 12.18** - Component-aware eigenvector selection

- **Why merge**: These tasks have significant overlap
- **Why third**: Less critical now that blobs (disconnected) cases are fixed
- **Expected outcome**: More robust handling of edge cases
- **Impact**: Improve reliability but may not fix current failures

### Priority 4: Ensure Reproducibility

**Task 12.20** - Complete random state propagation

- **Why fourth**: Important for debugging but not blocking current failures
- **Expected outcome**: Deterministic results across runs
- **Impact**: Makes debugging easier

### Priority 5: Deep Investigation (If Needed)

**Task 12.14** - Debug why sklearn achieves perfect ARI with 3 components for 2 clusters

- **Why fifth**: May be addressed by Task 12.19 findings
- **Expected outcome**: Understanding of sklearn's k-means clustering strategy
- **Impact**: Could be critical if Task 12.19 doesn't reveal the issue

### Deprioritized/Potentially Obsolete

**Task 12.16** - Investigate eigensolver differences

- **Why deprioritized**: Our findings show the eigensolver isn't the issue
- **Recommendation**: Mark as obsolete unless new evidence emerges

**Task 12.21** - Improve spectral clustering for overlapping clusters

- **Why last**: This is more of a general improvement task
- **Recommendation**: Defer until after achieving sklearn parity

## Recommended Next Steps

1. **Start with Task 12.19** - Create comprehensive sklearn comparison
   - Focus on k-NN graph structure differences
   - Check if sklearn does any graph preprocessing
   - Verify our diffusion scaling implementation matches exactly

2. **Quick win with Task 12.17** - Fix RBF gamma scaling
   - Should be straightforward to implement
   - Could immediately improve 2-3 test results

3. **Merge Tasks 12.15 & 12.18** - Create single component-handling task
   - Reduce duplication of effort
   - Focus on practical implementation

## Key Questions to Answer

1. Why does sklearn's first embedding dimension become constant for k-NN graphs when the constant vector is NOT an eigenvector?
2. Does sklearn preprocess the affinity matrix in some way?
3. Is there a different eigensolver mode or parameter we're missing?
4. Does sklearn handle the diffusion scaling differently for different eigenvalue ranges?

## Success Metrics

- **Short term**: Get to 6-7/12 tests passing by fixing RBF gamma
- **Medium term**: Get to 10/12 tests passing by solving k-NN issues
- **Long term**: Achieve 12/12 tests passing for full sklearn parity

## Risks and Mitigation

- **Risk**: The k-NN issue might be fundamental to our eigensolver
- **Mitigation**: Task 12.19 should reveal if we need a different approach

- **Risk**: We might be chasing numerical precision issues
- **Mitigation**: Focus on algorithmic differences first

## Conclusion

The path forward is clearer now that we understand sklearn uses diffusion map scaling. The priority should be understanding the k-NN graph structure issues through comprehensive comparison (Task 12.19), while picking up the quick win of fixing RBF gamma scaling (Task 12.17).
