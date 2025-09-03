---
id: task-33.1
title: Design SOM architecture and integration strategy
status: Done
assignee: []
created_date: '2025-09-02 21:36'
updated_date: '2025-09-02 21:43'
labels: []
dependencies: []
parent_task_id: task-33
---

## Description

Plan the overall architecture for SOM implementation, including how it integrates with existing clustering infrastructure, extends base interfaces, and follows library patterns. Define the module structure, class hierarchy, and data flow.

## Acceptance Criteria

- [x] Architecture design documented
- [x] Integration points identified
- [x] Module structure defined
- [x] Data flow diagrams created
- [x] API contracts specified

## Implementation Plan

1. Analyze existing clustering infrastructure patterns
2. Define SOM class hierarchy and interfaces
3. Map TensorFlow.js integration points
4. Design data flow for training and prediction
5. Specify API contracts following library conventions

## Implementation Notes

### Architecture Overview

The SOM implementation follows the established patterns in the clustering library:

#### 1. Class Hierarchy

```
BaseClustering<SOMParams> (interface)
    └── SOM (class)
         ├── Implements fit() and fitPredict()
         ├── Adds partialFit() for online learning
         ├── Extends with SOM-specific methods
         └── Uses TensorFlow.js for all computations
```

#### 2. Module Structure

```
src/
├── clustering/
│   ├── som.ts              # Main SOM class implementation
│   ├── som_utils.ts        # Helper functions (BMU, neighborhoods)
│   └── types.ts            # Extended with SOMParams interface
├── utils/
│   └── som_visualization.ts # Visualization utilities
└── index.ts                 # Export SOM class
```

#### 3. Integration Points

- **BaseClustering Interface**: SOM implements the standard interface with `fit()` and `fitPredict()`
- **TensorFlow.js Backend**: Uses `tf-adapter` for backend-agnostic tensor operations
- **Type System**: Extends `BaseClusteringParams` with `SOMParams`
- **Export System**: Added to main exports alongside KMeans, SpectralClustering, etc.

#### 4. Data Flow

**Training Flow:**
```
Input (DataMatrix) → Tensor2D conversion
    → Grid initialization (weights as Tensor3D)
    → For each epoch:
        → BMU calculation (tf.argMin of distances)
        → Neighborhood influence computation
        → Weight updates (tf operations)
    → Convergence check
    → Output: trained weights, labels
```

**Prediction Flow:**
```
Input (DataMatrix) → Tensor2D conversion
    → BMU calculation for each sample
    → Return BMU indices as labels
```

**Online Learning Flow:**
```
partialFit(batch) → Update existing weights
    → Track total samples for decay
    → Maintain learning state
    → Allow continuous refinement
```

#### 5. API Contracts

```typescript
interface SOMParams extends BaseClusteringParams {
  // Grid dimensions (replaces nClusters)
  gridWidth: number;
  gridHeight: number;
  
  // Core parameters
  topology?: 'rectangular' | 'hexagonal';
  neighborhood?: 'gaussian' | 'bubble' | 'mexican_hat';
  numEpochs?: number;
  learningRate?: number | ((epoch: number) => number);
  radius?: number | ((epoch: number) => number);
  
  // Initialization
  initialization?: 'random' | 'linear' | 'pca';
  
  // Online learning
  onlineMode?: boolean;
  miniBatchSize?: number;
}

class SOM implements BaseClustering<SOMParams> {
  // Standard clustering interface
  async fit(X: DataMatrix): Promise<void>;
  async fitPredict(X: DataMatrix): Promise<LabelVector>;
  
  // SOM-specific extensions
  async partialFit(X: DataMatrix): Promise<void>;
  predict(X: DataMatrix): Promise<LabelVector>;
  getWeights(): tf.Tensor3D;
  getUMatrix(): tf.Tensor2D;
  quantizationError(): number;
  topographicError(): number;
  
  // State management
  saveState(): SOMState;
  loadState(state: SOMState): void;
}
```

#### 6. TensorFlow.js Optimization Strategy

- **Batch Operations**: Process multiple samples simultaneously using tf.matMul
- **Memory Management**: Use tf.tidy() for automatic tensor cleanup
- **GPU Acceleration**: Leverage WebGL backend for distance calculations
- **Tensor Reuse**: Pre-allocate tensors for distances and updates
- **Efficient Broadcasting**: Use tf broadcasting for neighborhood updates

#### 7. Testing Strategy

- Unit tests for each component (grid, BMU, neighborhoods)
- Reference tests against MiniSom fixtures
- Memory leak tests using tf.memory()
- Performance benchmarks comparing CPU vs GPU
- Integration tests with existing clustering infrastructure

### Design Decisions

1. **Grid as Primary Structure**: Unlike traditional clusterers with `nClusters`, SOM uses `gridWidth` and `gridHeight` to define the map size. The total number of neurons is `gridWidth * gridHeight`.

2. **Tensor3D for Weights**: Weights stored as [height, width, features] for efficient grid operations and visualization.

3. **Separate Visualization Module**: Visualization utilities in separate module to keep core implementation focused.

4. **Online Learning as First-Class Feature**: `partialFit()` method for streaming data support, critical for large datasets.

5. **Decay Functions as Parameters**: Learning rate and radius can be functions for custom decay strategies.

### Next Steps

With the architecture defined, the implementation will proceed through:
- Task 33.2: Set up reference implementation for testing
- Task 33.3: Implement type definitions following this design
- Task 33.4+: Build components according to this architecture
