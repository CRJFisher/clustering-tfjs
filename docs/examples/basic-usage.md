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
  
  // Test k from 2 to 8
  const result = await findOptimalClusters(data, {
    minClusters: 2,
    maxClusters: 8,
    algorithm: 'kmeans'
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

## Next Steps

- Check out the [API Reference](../API.md) for detailed method documentation
- See [Performance Guide](../performance.md) for optimization tips
- Read [Migration Guide](../migration-guide.md) if coming from scikit-learn