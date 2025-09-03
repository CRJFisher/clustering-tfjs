# Basic Usage Examples

This guide provides practical examples for using clustering-tfjs in common scenarios.

## Installation

```bash
npm install clustering-tfjs
```

## Simple K-Means Example

```typescript
import { KMeans } from 'clustering-tfjs';

// Sample 2D data points
const data = [
  [1, 2], [1.5, 1.8], [5, 8], 
  [8, 8], [1, 0.6], [9, 11],
  [8, 2], [10, 2], [9, 3]
];

async function basicKMeans() {
  // Create KMeans instance with 3 clusters
  const kmeans = new KMeans({ nClusters: 3 });
  
  // Fit the model and get cluster labels
  const labels = await kmeans.fitPredict(data);
  
  console.log('Cluster labels:', labels);
  // Output: [0, 0, 1, 1, 0, 2, 2, 2, 2]
  
  // Access cluster centers
  console.log('Cluster centers:', await kmeans.centroids_.array());
}

basicKMeans();
```

## Finding Optimal Number of Clusters

```typescript
import { findOptimalClusters } from 'clustering-tfjs';

async function findBestK() {
  const data = [
    // ... your data points
  ];
  
  // Test k from 2 to 8 with K-means
  const result = await findOptimalClusters(data, {
    minClusters: 2,
    maxClusters: 8,
    algorithm: 'kmeans'
  });
  
  // Also works with SOM
  const somResult = await findOptimalClusters(data, {
    minClusters: 4,
    maxClusters: 16,
    algorithm: 'som',
    algorithmParams: {
      topology: 'hexagonal',
      initialization: 'pca'
    }
  });
  
  console.log(`Optimal number of clusters: ${result.optimal.k}`);
  console.log(`Silhouette score: ${result.optimal.silhouette.toFixed(3)}`);
  console.log(`Davies-Bouldin index: ${result.optimal.daviesBouldin.toFixed(3)}`);
  
  // See all tested configurations
  result.evaluations.forEach(eval => {
    console.log(`k=${eval.k}: silhouette=${eval.silhouette.toFixed(3)}`);
  });
}
```

## Working with Different Data Formats

```typescript
import { KMeans } from 'clustering-tfjs';
import * as tf from '@tensorflow/tfjs-node';

async function differentDataFormats() {
  // 1. Array of arrays (most common)
  const arrayData = [[1, 2], [3, 4], [5, 6]];
  
  // 2. TensorFlow.js tensor
  const tensorData = tf.tensor2d([[1, 2], [3, 4], [5, 6]]);
  
  const kmeans = new KMeans({ nClusters: 2 });
  
  // Both formats work the same way
  const labels1 = await kmeans.fitPredict(arrayData);
  const labels2 = await kmeans.fitPredict(tensorData);
  
  // Remember to dispose tensors when done
  tensorData.dispose();
}
```

## Spectral Clustering for Non-Convex Shapes

```typescript
import { SpectralClustering } from 'clustering-tfjs';

async function spectralExample() {
  // Generate two half-moons (non-convex shapes)
  const generateHalfMoon = (n: number, radius: number, noise: number, offset: number[]) => {
    const points = [];
    for (let i = 0; i < n; i++) {
      const angle = Math.PI * i / n;
      const x = radius * Math.cos(angle) + (Math.random() - 0.5) * noise + offset[0];
      const y = radius * Math.sin(angle) + (Math.random() - 0.5) * noise + offset[1];
      points.push([x, y]);
    }
    return points;
  };
  
  const moon1 = generateHalfMoon(100, 2, 0.3, [0, 0]);
  const moon2 = generateHalfMoon(100, 2, 0.3, [1.5, -0.5]);
  const data = [...moon1, ...moon2];
  
  // Spectral clustering handles non-convex shapes better
  const spectral = new SpectralClustering({
    nClusters: 2,
    affinity: 'rbf',
    gamma: 10  // Adjust based on data scale
  });
  
  const labels = await spectral.fitPredict(data);
  console.log('Successfully separated two half-moons');
}
```

## Hierarchical Clustering

```typescript
import { AgglomerativeClustering } from 'clustering-tfjs';

async function hierarchicalExample() {
  // Customer segmentation data
  const customerData = [
    [25, 50000],   // age, income
    [30, 55000],
    [35, 60000],
    [20, 20000],
    [25, 25000],
    [30, 30000],
    [60, 100000],
    [65, 110000],
    [70, 120000]
  ];
  
  // Normalize data for better results
  const normalize = (data: number[][]) => {
    const means = data[0].map((_, i) => 
      data.reduce((sum, row) => sum + row[i], 0) / data.length
    );
    const stds = data[0].map((_, i) => 
      Math.sqrt(data.reduce((sum, row) => 
        sum + Math.pow(row[i] - means[i], 2), 0) / data.length)
    );
    
    return data.map(row => 
      row.map((val, i) => (val - means[i]) / stds[i])
    );
  };
  
  const normalizedData = normalize(customerData);
  
  const clustering = new AgglomerativeClustering({
    nClusters: 3,
    linkage: 'ward'  // Minimizes within-cluster variance
  });
  
  const labels = await clustering.fitPredict(normalizedData);
  console.log('Customer segments:', labels);
}
```

## Evaluating Clustering Quality

```typescript
import { KMeans, silhouetteScore, daviesBouldin, calinskiHarabasz } from 'clustering-tfjs';

async function evaluateClustering() {
  const data = [
    // Your data points
  ];
  
  const kmeans = new KMeans({ nClusters: 3 });
  const labels = await kmeans.fitPredict(data);
  
  // Calculate all metrics
  const [silhouette, davies, calinski] = await Promise.all([
    silhouetteScore(data, labels),
    daviesBouldin(data, labels),
    calinskiHarabasz(data, labels)
  ]);
  
  console.log('Clustering Quality Metrics:');
  console.log(`Silhouette Score: ${silhouette.toFixed(3)} (higher is better, range: [-1, 1])`);
  console.log(`Davies-Bouldin Index: ${davies.toFixed(3)} (lower is better)`);
  console.log(`Calinski-Harabasz Index: ${calinski.toFixed(3)} (higher is better)`);
  
  // Interpret results
  if (silhouette > 0.5) {
    console.log('Good cluster separation');
  } else if (silhouette > 0.25) {
    console.log('Moderate cluster separation');
  } else {
    console.log('Poor cluster separation - consider different k or algorithm');
  }
}
```

## Handling Large Datasets

```typescript
import { KMeans } from 'clustering-tfjs';
import * as tf from '@tensorflow/tfjs-node';

async function largeDatesetClustering() {
  // For large datasets, use mini-batch approach
  const batchSize = 1000;
  const totalSamples = 100000;
  const features = 50;
  
  // Generate random data in batches
  const generateBatch = () => 
    Array.from({ length: batchSize }, () =>
      Array.from({ length: features }, () => Math.random())
    );
  
  // Use sampling for initial clustering
  const sampleSize = 5000;
  const sampleData = Array.from({ length: sampleSize }, () =>
    Array.from({ length: features }, () => Math.random())
  );
  
  const kmeans = new KMeans({ 
    nClusters: 10,
    maxIter: 100  // Reduce iterations for speed
  });
  
  // Fit on sample
  await kmeans.fit(sampleData);
  
  // Predict on batches
  for (let i = 0; i < totalSamples / batchSize; i++) {
    const batch = generateBatch();
    const labels = await kmeans.predict(batch);
    // Process labels...
  }
}
```

## Custom Distance Metrics (Using Spectral)

```typescript
import { SpectralClustering } from 'clustering-tfjs';

async function customAffinityExample() {
  // Time series data
  const timeSeries = [
    [1, 2, 3, 4, 5],
    [2, 3, 4, 5, 6],
    [5, 4, 3, 2, 1],
    [6, 5, 4, 3, 2],
    [1, 1, 5, 5, 5],
    [2, 2, 6, 6, 6]
  ];
  
  // Use spectral clustering with custom affinity
  const spectral = new SpectralClustering({
    nClusters: 3,
    affinity: 'rbf',
    gamma: 0.1  // Adjust for your data scale
  });
  
  const labels = await spectral.fitPredict(timeSeries);
  console.log('Time series clusters:', labels);
}
```

## Error Handling

```typescript
import { KMeans } from 'clustering-tfjs';

async function robustClustering() {
  const data = [[1, 2], [3, 4]];  // Only 2 points
  
  try {
    // This will fail - can't create 3 clusters from 2 points
    const kmeans = new KMeans({ nClusters: 3 });
    await kmeans.fitPredict(data);
  } catch (error) {
    console.error('Clustering failed:', error.message);
    
    // Fall back to fewer clusters
    const kmeans = new KMeans({ nClusters: 2 });
    const labels = await kmeans.fitPredict(data);
    console.log('Fallback clustering:', labels);
  }
}
```

## Self-Organizing Maps (SOM) Example

```typescript
import { SOM } from 'clustering-tfjs';

async function somExample() {
  // Generate sample 2D data with 3 distinct groups
  const data = [
    // Group 1 - bottom left
    [1, 2], [1.5, 1.8], [1.2, 2.3], [0.8, 1.7], [1.3, 2.1],
    // Group 2 - top right
    [8, 9], [8.5, 8.8], [8.2, 9.3], [7.8, 8.7], [8.3, 9.1],
    // Group 3 - top left
    [2, 8], [2.5, 7.8], [2.2, 8.3], [1.8, 7.7], [2.3, 8.1]
  ];
  
  // Create a 4x4 SOM grid
  const som = new SOM({
    gridWidth: 4,
    gridHeight: 4,
    nClusters: 16,
    topology: 'rectangular',  // or 'hexagonal'
    neighborhood: 'gaussian',
    initialization: 'pca',    // Use PCA for initialization
    learningRate: 0.5,
    radius: 2,
    numEpochs: 100,
    randomState: 42
  });
  
  // Train the SOM
  console.log('Training SOM...');
  const labels = await som.fitPredict(data);
  console.log('Cluster assignments:', labels);
  
  // Get the weight vectors (codebook)
  const weights = som.getWeights();
  console.log('SOM weights shape:', weights.shape); // [4, 4, 2]
  
  // Calculate U-matrix for visualization
  const uMatrix = som.getUMatrix();
  const uMatrixArray = await uMatrix.array();
  console.log('U-matrix (distances between neurons):');
  console.table(uMatrixArray);
  
  // Evaluate map quality
  const quantError = som.quantizationError();
  const topoError = await som.topographicError(data);
  console.log(`Quantization error: ${quantError.toFixed(4)}`);
  console.log(`Topographic error: ${topoError.toFixed(4)}`);
  
  // Clean up tensors
  weights.dispose();
  uMatrix.dispose();
}

somExample();
```

### SOM for Data Visualization

```typescript
import { SOM } from 'clustering-tfjs';

async function somVisualization() {
  // High-dimensional data example (e.g., features extracted from images)
  const highDimData = generateHighDimensionalData(); // Your data here
  
  // Create a larger SOM for better visualization
  const som = new SOM({
    gridWidth: 10,
    gridHeight: 10,
    nClusters: 100,
    topology: 'hexagonal',    // Hexagonal grid for smoother transitions
    neighborhood: 'gaussian',
    initialization: 'pca',
    numEpochs: 200,           // More epochs for complex data
    learningRate: 0.7,        // Higher initial learning rate
    radius: 5                 // Larger initial radius
  });
  
  // Train the SOM
  await som.fit(highDimData);
  
  // Get BMU (Best Matching Unit) positions for each data point
  const labels = await som.predict(highDimData);
  
  // Convert flat labels to 2D grid positions
  const gridPositions = labels.map(label => {
    const row = Math.floor(label / som.params.gridWidth);
    const col = label % som.params.gridWidth;
    return [row, col];
  });
  
  // Use grid positions for 2D visualization
  console.log('2D positions for visualization:', gridPositions);
  
  // Create U-matrix for understanding cluster boundaries
  const uMatrix = som.getUMatrix();
  
  // U-matrix shows distances between adjacent neurons
  // High values indicate cluster boundaries
  // Low values indicate similar regions
  
  return { gridPositions, uMatrix };
}
```

### Incremental/Online Learning with SOM

```typescript
import { SOM } from 'clustering-tfjs';

async function onlineLearning() {
  // Create SOM for streaming data
  const som = new SOM({
    gridWidth: 10,
    gridHeight: 10,
    nClusters: 100,
    initialization: 'random',
    learningRate: 0.5,
    numEpochs: 10  // Few epochs per batch
  });
  
  // Process data in batches (simulating streaming)
  const batches = [
    generateBatch1(), // First batch of data
    generateBatch2(), // Second batch
    generateBatch3()  // Third batch
  ];
  
  for (let i = 0; i < batches.length; i++) {
    console.log(`Processing batch ${i + 1}...`);
    
    // Continue training with new data
    await som.fit(batches[i]);
    
    // Monitor convergence
    const quantError = som.quantizationError();
    console.log(`Quantization error after batch ${i + 1}: ${quantError.toFixed(4)}`);
  }
  
  // Final evaluation
  const allData = batches.flat();
  const finalError = await som.topographicError(allData);
  console.log(`Final topographic error: ${finalError.toFixed(4)}`);
}
```

### Using SOM with Different Topologies

```typescript
import { SOM } from 'clustering-tfjs';

async function compareTopologies() {
  const data = generateYourData(); // Your data
  
  // Rectangular topology (4 or 8 neighbors)
  const rectSom = new SOM({
    gridWidth: 5,
    gridHeight: 5,
    nClusters: 25,
    topology: 'rectangular',
    initialization: 'pca'
  });
  
  // Hexagonal topology (6 neighbors) - better for visualization
  const hexSom = new SOM({
    gridWidth: 5,
    gridHeight: 5,
    nClusters: 25,
    topology: 'hexagonal',
    initialization: 'pca'
  });
  
  // Train both
  await rectSom.fit(data);
  await hexSom.fit(data);
  
  // Compare topographic errors
  const rectError = await rectSom.topographicError(data);
  const hexError = await hexSom.topographicError(data);
  
  console.log(`Rectangular topology error: ${rectError.toFixed(4)}`);
  console.log(`Hexagonal topology error: ${hexError.toFixed(4)}`);
  
  // Lower topographic error indicates better topology preservation
  if (hexError < rectError) {
    console.log('Hexagonal topology preserves data topology better');
  } else {
    console.log('Rectangular topology preserves data topology better');
  }
}
```

## Next Steps

- Check out the [API Reference](../API.md) for detailed method documentation
- See [Performance Guide](../performance.md) for optimization tips
- Read [Migration Guide](../migration-guide.md) if coming from scikit-learn
