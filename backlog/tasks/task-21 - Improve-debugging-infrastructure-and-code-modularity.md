---
id: task-21
title: Improve debugging infrastructure and code modularity
status: To Do
assignee: []
created_date: '2025-07-21'
labels:
  - infrastructure
  - debugging
  - refactoring
dependencies: []
---

## Description

Following the discovery of a critical issue where tests were passing with an outdated implementation (src/*.js files) while the actual distributed code was failing, we need to improve our debugging infrastructure and code modularity. This task addresses systematic improvements to prevent similar issues and make the codebase more maintainable and debuggable.

## Acceptance Criteria

- [ ] Remove all compiled JS files from git tracking and update .gitignore
- [ ] Refactor SpectralClustering to expose intermediate computation steps
- [ ] Create organized debugging tools directory structure
- [ ] Implement debug mode for capturing intermediate results
- [ ] Add integration tests that verify individual algorithmic steps
- [ ] Document debugging procedures and tools
- [ ] Ensure consistent import paths in all tests

## Implementation Plan

### 1. Fix Version Control Issues

1. Remove all `src/**/*.js` files from git
2. Update `.gitignore` to exclude:
   ```
   src/**/*.js
   src/**/*.d.ts
   src/**/*.js.map
   ```
3. Add pre-commit hook to prevent accidental commits of compiled files
4. Update build process documentation

### 2. Refactor for Modularity

Refactor `SpectralClustering` class to expose intermediate steps:

```typescript
export class SpectralClustering {
  // Expose intermediate steps as public methods
  computeAffinityMatrix(X: DataMatrix): tf.Tensor2D
  computeLaplacian(affinity: tf.Tensor2D): LaplacianResult
  computeSpectralEmbedding(laplacian: tf.Tensor2D): EmbeddingResult
  performClustering(embedding: tf.Tensor2D): number[]
  
  // Main pipeline orchestrates the steps
  fitPredict(X: DataMatrix): number[]
}
```

This enables:
- Testing individual steps in isolation
- Comparing intermediate results with reference implementations
- Debugging specific transformations
- Reusing components in other algorithms

### 3. Organize Debugging Infrastructure

Create directory structure:
```
tools/
├── debug/
│   ├── compare_implementations.py
│   ├── visualize_clustering.py
│   └── analyze_eigenvectors.py
├── sklearn_comparison/
│   ├── generate_intermediates.py
│   ├── compare_step_by_step.py
│   └── fixtures/
└── README.md
```

### 4. Implement Debug Mode

Add optional debug capture to algorithms:

```typescript
interface DebugInfo {
  affinityMatrix?: MatrixStats;
  laplacianSpectrum?: number[];
  embeddingStats?: EmbeddingStats;
  clusteringMetrics?: ClusteringMetrics;
}

export class SpectralClustering {
  constructor(params: SpectralParams & { captureDebugInfo?: boolean }) {
    this.captureDebugInfo = params.captureDebugInfo ?? false;
  }
  
  getDebugInfo(): DebugInfo | undefined {
    return this.debugInfo;
  }
}
```

### 5. Create Integration Tests

Add step-by-step comparison tests:

```typescript
// test/integration/spectral_steps.test.ts
describe('SpectralClustering step verification', () => {
  test.each(fixtures)('affinity matrix computation - %s', async (fixtureName) => {
    const { X, expectedAffinity } = loadFixture(fixtureName);
    const spectral = new SpectralClustering(params);
    const affinity = await spectral.computeAffinityMatrix(X);
    
    expect(affinity).toMatchMatrixStatistics(expectedAffinity);
  });
  
  // Similar tests for each step...
});
```

### 6. Document Debugging Procedures

Create `docs/debugging-guide.md`:
- How to use debug scripts
- Common issues and solutions
- How to capture sklearn intermediates
- How to add new comparison tests

## Background and Lessons Learned

### The Incident

During task 12.14, we discovered that:
1. Compiled JS files (`src/*.js`) were tracked in git alongside TypeScript sources
2. Tests imported from `src/` and used old implementations
3. Production code in `dist/` had different implementations
4. This created false test passes - tests showed 5/12 passing but reality was worse

### Key Lessons

1. **Never commit generated files** - Only source files belong in version control
2. **Single source of truth** - Tests and production must use the same code
3. **Diagnostic scripts are valuable** - But need organization and documentation
4. **Monolithic implementations hide bugs** - Modular code is easier to test and debug
5. **Intermediate results matter** - Being able to inspect each step is crucial

### Benefits of This Task

- Prevents similar dual-implementation bugs
- Makes debugging systematic rather than ad-hoc
- Enables fine-grained testing of algorithmic steps
- Provides reusable tools for future debugging
- Improves code quality through modularity

## Technical Debt Addressed

- Removes confusion about which code is actually running
- Eliminates the src/ vs dist/ discrepancy
- Provides infrastructure for systematic sklearn comparison
- Makes the codebase more maintainable
- Reduces debugging time for future issues