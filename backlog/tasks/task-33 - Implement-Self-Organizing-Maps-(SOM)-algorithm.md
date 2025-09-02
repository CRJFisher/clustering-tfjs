---
id: task-33
title: Implement Self-Organizing Maps (SOM) algorithm
status: To Do
assignee: []
created_date: '2025-09-02 14:40'
labels: []
dependencies: []
---

## Description

Add Self-Organizing Maps (Kohonen maps) implementation for unsupervised learning and data visualization. SOMs create a low-dimensional (typically 2D) discrete representation of high-dimensional input space while preserving topological properties, making them valuable for exploratory data analysis, dimensionality reduction, and clustering visualization.

## Acceptance Criteria

- [ ] SOM algorithm implemented with configurable grid topology (rectangular and hexagonal)
- [ ] Neighborhood functions implemented (Gaussian, bubble, Mexican hat)
- [ ] Learning rate and radius decay strategies implemented
- [ ] Best Matching Unit (BMU) search optimized with TensorFlow.js operations
- [ ] Weight initialization strategies (random, linear, PCA-based)
- [ ] Training convergence monitoring and early stopping
- [ ] Online/incremental training support for streaming data
- [ ] Reference implementation comparison tests passing
- [ ] Visualization utilities for trained maps
- [ ] Performance benchmarks documented
- [ ] API documentation with usage examples

## Implementation Plan

### 1. Research Phase
- Study SOM algorithm theory and mathematical foundations
- Identify reference implementation (MiniSom or sklearn-som Python library)
- Document key parameters and their effects on training
- Research TensorFlow.js optimization strategies for SOM operations

### 2. Core Algorithm Implementation

**Create `src/clustering/som.ts`**:
```typescript
interface SOMOptions {
  gridWidth: number;
  gridHeight: number;
  topology: 'rectangular' | 'hexagonal';
  neighborhood: 'gaussian' | 'bubble' | 'mexican_hat';
  learningRate: number | ((epoch: number) => number);
  radius: number | ((epoch: number) => number);
  numEpochs: number;
  initialization: 'random' | 'linear' | 'pca';
  randomState?: number;
  onlineMode?: boolean;  // Enable incremental training
  miniBatchSize?: number;  // For mini-batch online learning
}
```

**Key components**:
- Grid initialization and topology management
- BMU calculation using efficient tensor operations
- Neighborhood function implementations
- Weight update mechanism
- Training loop with batch processing support
- Online/incremental learning state management

### 3. Mathematical Operations

**BMU Selection** (vectorized):
```typescript
// Compute distances from input to all neurons
// distances = ||x - weights||² for each neuron
// bmu = argmin(distances)
```

**Neighborhood Update**:
```typescript
// h(distance, radius) = exp(-distance² / (2 * radius²))
// Δweight = learningRate * h * (input - weight)
```

### 4. Reference Implementation Testing

**Create `test/clustering/som.reference.test.ts`**:
- Generate test fixtures using MiniSom or sklearn-som
- Test various configurations:
  - Small dataset (iris: 150×4)
  - Medium dataset (digits: 1797×64)
  - Different grid sizes (5×5, 10×10, 20×20)
  - Different topologies and neighborhoods
- Validate:
  - Final weight matrices
  - BMU assignments
  - Quantization error
  - Topographic error

### 5. Optimization Strategies

- **Batch BMU computation**: Process multiple samples simultaneously
- **Tensor reuse**: Pre-allocate tensors for distances and updates
- **GPU acceleration**: Leverage tf.matMul for distance calculations
- **Sparse updates**: Only update neurons within radius threshold
- **Early stopping**: Monitor quantization error convergence

### 6. Visualization Utilities

**Create `src/utils/som_visualization.ts`**:
- U-matrix (unified distance matrix) computation
- Component planes visualization
- Hit map generation
- BMU trajectory tracking
- Export utilities for external visualization tools

### 7. API Design

```typescript
class SOM extends BaseClusterer {
  constructor(options: SOMOptions);
  
  async fit(X: Tensor2D): Promise<void>;
  async partialFit(X: Tensor2D): Promise<void>;  // Incremental training
  predict(X: Tensor2D): Promise<Tensor1D>;  // Returns BMU indices
  getWeights(): Tensor3D;  // [height, width, features]
  getUMatrix(): Tensor2D;  // Unified distance matrix
  quantizationError(): number;
  topographicError(): number;
  getTotalSamplesLearned(): number;  // Track incremental learning progress
  saveState(): SOMState;  // Serialize for persistence
  loadState(state: SOMState): void;  // Resume from saved state
}
```

### 8. Online/Incremental Learning

**Implementation approach**:
- Maintain learning state between `partialFit` calls
- Track total samples seen for proper decay scheduling
- Support both single-sample and mini-batch updates
- Preserve neighborhood and learning rate schedules across sessions

**State management**:
```typescript
interface SOMState {
  weights: number[][][];
  totalSamples: number;
  currentEpoch: number;
  learningRateSchedule: LearningSchedule;
  radiusSchedule: RadiusSchedule;
}
```

**Use cases**:
- Real-time data streams
- Large datasets that don't fit in memory
- Adaptive learning for changing data distributions
- Continuous model refinement

## Technical Notes

### Algorithm Details

1. **Grid Topology**:
   - Rectangular: Simple 2D grid with 4 or 8 neighbors
   - Hexagonal: Each neuron has 6 neighbors, better for visualization

2. **Neighborhood Functions**:
   - **Gaussian**: h(d,σ) = exp(-d²/2σ²), smooth decay
   - **Bubble**: h(d,σ) = 1 if d≤σ else 0, hard cutoff
   - **Mexican Hat**: h(d,σ) = (1-d²/σ²)exp(-d²/2σ²), lateral inhibition

3. **Decay Strategies**:
   - Linear: param(t) = param₀ × (1 - t/T)
   - Exponential: param(t) = param₀ × exp(-t/τ)
   - Inverse: param(t) = param₀ / (1 + t/T)

4. **Initialization Methods**:
   - Random: Uniform or normal distribution
   - Linear: Along first two principal components
   - PCA: Using principal component analysis

### Performance Considerations

- Typical training: O(n × m × k × epochs) where n=samples, m=neurons, k=features
- Memory: O(m × k) for weights, O(n × m) for distance matrix if not batched
- GPU benefits most from large grids and high-dimensional data
- Consider mini-batch training for very large datasets
- Online learning reduces memory to O(m × k + b) where b=batch size
- Incremental updates allow processing infinite streams

### Reference Implementation

Since scikit-learn doesn't include SOM, use **MiniSom** (Python):
```bash
pip install minisom
```

Example fixture generation:
```python
from minisom import MiniSom
import numpy as np

# Create and train SOM
som = MiniSom(10, 10, n_features, sigma=1.0, learning_rate=0.5)
som.train(data, num_epochs)

# Export for testing
weights = som.get_weights()
bmus = np.array([som.winner(x) for x in data])
```

### Integration Points

- Extends BaseClusterer for consistency with other algorithms
- Uses existing TensorFlow.js backend infrastructure
- Follows library patterns for options and validation
- Integrates with existing test framework
