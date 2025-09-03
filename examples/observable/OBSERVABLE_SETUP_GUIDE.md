# Observable Setup Guide for clustering-tfjs

This guide will walk you through setting up interactive clustering visualizations on Observable.

## Step 1: Create an Observable Account

1. Go to [observablehq.com](https://observablehq.com)
2. Click "Sign up" (it's free)
3. You can sign up with GitHub, Google, or email

## Step 2: Create Your First Notebook

### Option A: Quick Start (Recommended)

1. Once logged in, click the "+" button or "New" to create a notebook
2. Copy and paste each of these code blocks into separate cells:

**Cell 1 - Load Libraries:**
```javascript
tf = require("https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.20.0/dist/tf.min.js")
ClusteringTFJS = require("https://unpkg.com/clustering-tfjs@0.3.1/dist/clustering.browser.js")
```

**Cell 2 - Initialize Clustering:**
```javascript
clustering = {
  const { Clustering } = ClusteringTFJS;
  await Clustering.init({ backend: 'cpu' });
  return Clustering;
}
```

**Cell 3 - Generate Sample Data:**
```javascript
data = {
  const points = [];
  // Generate 3 clusters
  for (let i = 0; i < 3; i++) {
    const cx = i * 4 + Math.random() * 2;
    const cy = i * 3 + Math.random() * 2;
    for (let j = 0; j < 50; j++) {
      points.push([
        cx + (Math.random() - 0.5) * 2,
        cy + (Math.random() - 0.5) * 2
      ]);
    }
  }
  return points;
}
```

**Cell 4 - Run K-Means Clustering:**
```javascript
clusters = {
  const kmeans = new clustering.KMeans({ nClusters: 3 });
  const labels = await kmeans.fitPredict(data);
  return data.map((point, i) => ({
    x: point[0],
    y: point[1],
    cluster: labels[i]
  }));
}
```

**Cell 5 - Visualize Results:**
```javascript
Plot.plot({
  marks: [
    Plot.dot(clusters, {
      x: "x",
      y: "y",
      fill: "cluster",
      r: 5
    })
  ],
  color: {
    type: "categorical",
    scheme: "tableau10"
  },
  width: 600,
  height: 400,
  grid: true
})
```

### Option B: Interactive Version with Controls

For a more interactive experience, add these cells:

**Cell 6 - Add Controls:**
```javascript
viewof nClusters = Inputs.range([2, 10], {step: 1, value: 3, label: "Number of clusters"})
viewof algorithm = Inputs.select(["kmeans", "spectral", "agglomerative"], {label: "Algorithm"})
```

**Cell 7 - Dynamic Clustering:**
```javascript
dynamicClusters = {
  let model;
  switch(algorithm) {
    case "kmeans":
      model = new clustering.KMeans({ nClusters });
      break;
    case "spectral":
      model = new clustering.SpectralClustering({ nClusters });
      break;
    case "agglomerative":
      model = new clustering.AgglomerativeClustering({ nClusters });
      break;
  }
  
  const labels = await model.fitPredict(data);
  return data.map((point, i) => ({
    x: point[0],
    y: point[1],
    cluster: labels[i],
    algorithm: algorithm
  }));
}
```

**Cell 8 - Update Visualization:**
```javascript
Plot.plot({
  marks: [
    Plot.dot(dynamicClusters, {
      x: "x",
      y: "y",
      fill: "cluster",
      r: 5,
      tip: true
    })
  ],
  color: {
    type: "categorical",
    scheme: "tableau10"
  },
  width: 600,
  height: 400,
  grid: true,
  caption: `${algorithm} clustering with ${nClusters} clusters`
})
```

## Step 3: Converting HTML Examples to Observable

To convert the HTML examples to Observable notebooks:

### 1. Extract JavaScript Logic
From the HTML files, copy the JavaScript logic (without the HTML structure).

### 2. Break Into Cells
Split the code into logical cells:
- Data loading/generation
- Algorithm initialization
- Clustering execution
- Visualization

### 3. Use Observable Features

**Replace DOM manipulation:**
```javascript
// HTML version
document.getElementById('result').innerHTML = labels;

// Observable version
labels // Just return the value, Observable displays it
```

**Replace event handlers with reactive cells:**
```javascript
// HTML version
document.getElementById('slider').addEventListener('change', (e) => {
  nClusters = e.target.value;
  runClustering();
});

// Observable version
viewof nClusters = Inputs.range([2, 10], {value: 3})
// Other cells automatically re-run when nClusters changes
```

**Use Observable Plot instead of D3/Chart.js:**
```javascript
// Instead of complex D3 code
Plot.plot({
  marks: [
    Plot.dot(data, {x: "x", y: "y", fill: "cluster"})
  ]
})
```

## Step 4: Advanced Features

### Adding Different Datasets

```javascript
viewof dataset = Inputs.select(["blobs", "moons", "circles"], {label: "Dataset type"})

data = {
  switch(dataset) {
    case "blobs":
      return generateBlobs();
    case "moons":
      return generateMoons();
    case "circles":
      return generateCircles();
  }
}

function generateMoons() {
  const points = [];
  const n = 100;
  
  // Upper moon
  for (let i = 0; i < n/2; i++) {
    const angle = Math.PI * i / (n/2);
    points.push([
      Math.cos(angle) + (Math.random() - 0.5) * 0.1,
      Math.sin(angle) + (Math.random() - 0.5) * 0.1
    ]);
  }
  
  // Lower moon
  for (let i = 0; i < n/2; i++) {
    const angle = Math.PI * i / (n/2);
    points.push([
      1 - Math.cos(angle) + (Math.random() - 0.5) * 0.1,
      0.5 - Math.sin(angle) + (Math.random() - 0.5) * 0.1
    ]);
  }
  
  return points;
}
```

### Performance Metrics

```javascript
metrics = {
  const startTime = performance.now();
  const labels = await model.fitPredict(data);
  const endTime = performance.now();
  
  return {
    time: (endTime - startTime).toFixed(2) + " ms",
    silhouette: await clustering.metrics.silhouetteScore(data, labels),
    daviesBouldin: await clustering.metrics.daviesBouldin(data, labels)
  };
}
```

### Animation for K-Means

```javascript
// Store iterations
kmeans = new clustering.KMeans({ 
  nClusters: 3,
  maxIter: 1  // Run one iteration at a time
});

// Animate
iterations = {
  const frames = [];
  const model = new clustering.KMeans({ nClusters: 3 });
  
  for (let i = 0; i < 10; i++) {
    model.maxIter = i + 1;
    const labels = await model.fitPredict(data);
    const centroids = await model.clusterCenters_;
    
    frames.push({
      iteration: i,
      labels: labels,
      centroids: centroids
    });
  }
  
  return frames;
}

// Use Observable's Scrubber for animation control
viewof frame = Scrubber(iterations, {format: d => `Iteration ${d.iteration}`})
```

## Step 5: Publishing Your Notebook

1. Click the "Publish" button in the top right
2. Choose visibility (public or private)
3. Add a title and description
4. Your notebook gets a permanent URL to share

## Tips and Best Practices

1. **Use Reactive Programming**: Let Observable handle updates automatically
2. **Keep Cells Small**: One concept per cell makes debugging easier
3. **Add Markdown**: Use markdown cells to explain your analysis
4. **Import Other Notebooks**: Reuse code with `import {cellName} from "@username/notebook"`
5. **Use Observable Inputs**: Built-in controls are better than custom HTML

## Example Complete Notebook Structure

```
# Clustering.js Demo

This notebook demonstrates machine learning clustering algorithms running entirely in the browser.

## Load Libraries
[Cell with require statements]

## Initialize
[Cell with initialization]

## Controls
[Cell with input controls]

## Generate Data
[Cell with data generation]

## Run Clustering
[Cell with clustering execution]

## Visualize Results
[Cell with Plot.plot]

## Performance Metrics
[Cell with metrics]
```

## Troubleshooting

### Common Issues:

1. **"Cannot find module"**: Make sure you're using the exact CDN URLs
2. **"tf is not defined"**: Load TensorFlow.js before clustering-tfjs
3. **Performance issues**: Use 'cpu' backend for better compatibility
4. **Large datasets**: Limit to ~1000 points for smooth interaction

### Getting Help:

- Observable Forums: [talk.observablehq.com](https://talk.observablehq.com)
- Observable Documentation: [observablehq.com/@observablehq/documentation](https://observablehq.com/@observablehq/documentation)
- clustering-tfjs Issues: [github.com/CRJFisher/clustering-tfjs/issues](https://github.com/CRJFisher/clustering-tfjs/issues)

## Share Your Creations!

Once you've created a notebook, share it with the community:
- Tweet with #ObservableHQ
- Submit to Observable's community highlights
- Open a PR to add your notebook link to our examples