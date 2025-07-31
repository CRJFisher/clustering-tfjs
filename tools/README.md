# Debugging and Comparison Tools

This directory contains tools for debugging, analyzing, and comparing our clustering implementations with reference implementations like scikit-learn.

## Directory Structure

```
tools/
├── debug/                      # General debugging and analysis scripts
│   ├── compare_implementations.py    # Compare our implementation with sklearn step-by-step
│   ├── visualize_clustering.py       # Visualize clustering results and analyze patterns
│   └── analyze_eigenvectors.py       # Analyze eigenvector properties and issues
│
├── sklearn_comparison/         # Sklearn-specific comparison tools
│   ├── compare_step_by_step.py      # Detailed comparison with sklearn's exact behavior
│   ├── generate_intermediates.py    # Generate intermediate results from sklearn (TODO)
│   └── fixtures/                    # Stored comparison results and test data
│
└── sklearn_fixtures/          # Existing fixture generation for tests
    ├── generate_spectral.py   # Generate spectral clustering test fixtures
    └── requirements.txt       # Python dependencies
```

## Usage

### Debug Scripts

All debug scripts should be run from the project root with the sklearn virtual environment activated:

```bash
# Activate the sklearn environment
source tools/sklearn_fixtures/.venv/bin/activate

# Run comparison analysis
python tools/debug/compare_implementations.py

# Visualize clustering results
python tools/debug/visualize_clustering.py

# Analyze eigenvector issues
python tools/debug/analyze_eigenvectors.py
```

### Sklearn Comparison

To perform detailed step-by-step comparison with sklearn:

```bash
# Run comprehensive comparison on all fixtures
python tools/sklearn_comparison/compare_step_by_step.py

# Results are saved to tools/sklearn_comparison/fixtures/
```

## Key Findings

These tools were instrumental in discovering critical issues:

1. **Dual Implementation Bug**: Tests were using outdated `src/*.js` files while production used `dist/*.js`
2. **Eigenvector Recovery vs Diffusion Scaling**: The old implementation used eigenvector recovery which worked better for disconnected components
3. **3-Components-2-Clusters Problem**: k-NN graphs can create more connected components than desired clusters

## Adding New Debug Tools

When adding new debug scripts:

1. Place general debugging tools in `tools/debug/`
2. Place sklearn-specific tools in `tools/sklearn_comparison/`
3. Document the tool's purpose and usage
4. Ensure the tool can be run from the project root
5. Use relative paths for file access

## Common Debugging Patterns

### Comparing Intermediate Results

```python
# Example: Compare affinity matrices
our_affinity = run_our_implementation(data)
sklearn_affinity = compute_sklearn_affinity(data)
compare_matrices(our_affinity, sklearn_affinity)
```

### Visualizing Algorithmic Steps

```python
# Example: Visualize eigenspace embedding
embedding = compute_embedding(laplacian)
plot_embedding_2d(embedding, labels)
```

### Capturing Debug Information

Use the new `captureDebugInfo` flag in SpectralClustering:

```typescript
const spectral = new SpectralClustering({
  nClusters: 2,
  affinity: 'rbf',
  captureDebugInfo: true
});

await spectral.fit(X);
const debugInfo = spectral.getDebugInfo();
console.log(debugInfo);
```

Or use the `fitWithIntermediateSteps` method to get all intermediate results:

```typescript
const spectral = new SpectralClustering({
  nClusters: 2,
  affinity: 'rbf'
});

const intermediateSteps = await spectral.fitWithIntermediateSteps(X);
// intermediateSteps contains: affinity, laplacian, embedding, labels
```