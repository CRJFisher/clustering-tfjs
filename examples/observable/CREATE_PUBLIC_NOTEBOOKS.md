# Creating Public Observable Notebooks for clustering-tfjs

This guide shows how to create publicly viewable Observable notebooks for the clustering-tfjs examples.

## Option 1: Create Notebooks Under Your Account

Since we can't create notebooks under an organization account without permissions, here's how to create them under your personal Observable account:

### Step 1: Sign in to Observable
1. Go to [observablehq.com](https://observablehq.com)
2. Sign in with your account

### Step 2: Create the Notebooks

Create these four notebooks by clicking "New" and pasting the code:

#### Notebook 1: Interactive Clustering Visualization

**Title:** "clustering-tfjs: Interactive Visualization Demo"

**Description:** "Explore K-Means, Spectral, and Agglomerative clustering algorithms with real-time parameter controls and multiple datasets."

**Code cells:**

```javascript
// Load libraries
tf = require("https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.20.0/dist/tf.min.js")
ClusteringTFJS = require("https://unpkg.com/clustering-tfjs@latest/dist/clustering.browser.js")
d3 = require("d3@7")
```

```javascript
// Initialize
clustering = {
  const { Clustering } = ClusteringTFJS;
  await Clustering.init({ backend: 'cpu' });
  return Clustering;
}
```

```javascript
// Controls
viewof algorithm = Inputs.select(["kmeans", "spectral", "agglomerative"], {
  label: "Algorithm",
  value: "kmeans"
})

viewof nClusters = Inputs.range([2, 10], {
  step: 1, 
  value: 3, 
  label: "Number of clusters"
})

viewof dataset = Inputs.select(["blobs", "moons", "circles", "anisotropic"], {
  label: "Dataset",
  value: "blobs"
})
```

```javascript
// Dataset generators
function generateDataset(type, nPoints = 150) {
  switch(type) {
    case "blobs":
      return generateBlobs(nClusters, nPoints);
    case "moons":
      return generateMoons(nPoints);
    case "circles":
      return generateCircles(nPoints);
    case "anisotropic":
      return generateAnisotropic(nPoints);
  }
}

function generateBlobs(k, n) {
  const points = [];
  const pointsPerCluster = Math.floor(n / k);
  
  for (let i = 0; i < k; i++) {
    const cx = (i % 3) * 4 + Math.random() * 2;
    const cy = Math.floor(i / 3) * 4 + Math.random() * 2;
    
    for (let j = 0; j < pointsPerCluster; j++) {
      points.push([
        cx + (Math.random() - 0.5) * 1.5,
        cy + (Math.random() - 0.5) * 1.5
      ]);
    }
  }
  return points;
}

function generateMoons(n) {
  const points = [];
  const half = Math.floor(n / 2);
  
  // Upper moon
  for (let i = 0; i < half; i++) {
    const angle = Math.PI * i / half;
    points.push([
      Math.cos(angle) * 2 + (Math.random() - 0.5) * 0.2,
      Math.sin(angle) * 2 + (Math.random() - 0.5) * 0.2
    ]);
  }
  
  // Lower moon
  for (let i = 0; i < half; i++) {
    const angle = Math.PI * i / half;
    points.push([
      2 - Math.cos(angle) * 2 + (Math.random() - 0.5) * 0.2,
      1 - Math.sin(angle) * 2 + (Math.random() - 0.5) * 0.2
    ]);
  }
  
  return points;
}

function generateCircles(n) {
  const points = [];
  const third = Math.floor(n / 3);
  
  // Inner circle
  for (let i = 0; i < third; i++) {
    const angle = 2 * Math.PI * i / third;
    const r = 1 + (Math.random() - 0.5) * 0.2;
    points.push([
      Math.cos(angle) * r,
      Math.sin(angle) * r
    ]);
  }
  
  // Outer circle
  for (let i = 0; i < n - third; i++) {
    const angle = 2 * Math.PI * i / (n - third);
    const r = 3 + (Math.random() - 0.5) * 0.3;
    points.push([
      Math.cos(angle) * r,
      Math.sin(angle) * r
    ]);
  }
  
  return points;
}

function generateAnisotropic(n) {
  const points = [];
  const third = Math.floor(n / 3);
  
  // Elongated cluster
  for (let i = 0; i < third; i++) {
    points.push([
      (Math.random() - 0.5) * 6,
      (Math.random() - 0.5) * 0.5
    ]);
  }
  
  // Round cluster
  for (let i = 0; i < third; i++) {
    const angle = 2 * Math.PI * Math.random();
    const r = Math.random() * 1.5;
    points.push([
      Math.cos(angle) * r - 3,
      Math.sin(angle) * r + 3
    ]);
  }
  
  // Dense cluster
  for (let i = 0; i < n - 2 * third; i++) {
    points.push([
      3 + (Math.random() - 0.5) * 0.8,
      3 + (Math.random() - 0.5) * 0.8
    ]);
  }
  
  return points;
}
```

```javascript
// Generate data
data = generateDataset(dataset)
```

```javascript
// Run clustering
result = {
  const startTime = performance.now();
  
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
  const endTime = performance.now();
  
  // Calculate metrics
  const silhouette = await clustering.metrics.silhouetteScore(data, labels);
  
  return {
    points: data.map((point, i) => ({
      x: point[0],
      y: point[1],
      cluster: labels[i]
    })),
    time: endTime - startTime,
    silhouette: silhouette,
    algorithm: algorithm,
    nClusters: nClusters
  };
}
```

```javascript
// Visualize
Plot.plot({
  title: `${algorithm.charAt(0).toUpperCase() + algorithm.slice(1)} Clustering`,
  subtitle: `Time: ${result.time.toFixed(1)}ms | Silhouette: ${result.silhouette.toFixed(3)}`,
  marks: [
    Plot.dot(result.points, {
      x: "x",
      y: "y",
      fill: "cluster",
      r: 6,
      opacity: 0.8,
      tip: true
    })
  ],
  color: {
    type: "categorical",
    scheme: "observable10"
  },
  width: 640,
  height: 400,
  grid: true,
  aspectRatio: 1
})
```

#### Notebook 2: K-Means Step-by-Step Animation

**Title:** "clustering-tfjs: K-Means Algorithm Animation"

**Description:** "Watch K-Means clustering algorithm in action with step-by-step visualization of centroid updates."

[Include animation code similar to kmeans-animation.html]

#### Notebook 3: Algorithm Comparison Dashboard

**Title:** "clustering-tfjs: Algorithm Comparison"

**Description:** "Compare K-Means, Spectral, and Agglomerative clustering side-by-side on the same dataset."

[Include comparison code similar to algorithm-comparison.html]

#### Notebook 4: Quick Start Template

**Title:** "clustering-tfjs: Quick Start"

**Description:** "Simple template to get started with clustering-tfjs in Observable."

[Include the basic 5-cell example from the guide]

### Step 3: Publish the Notebooks

1. For each notebook, click "Publish" in the top right
2. Make sure "Public" is selected
3. Add tags: `machine-learning`, `clustering`, `tensorflowjs`, `visualization`
4. Click "Publish notebook"

### Step 4: Get the Public URLs

After publishing, each notebook will have a URL like:
```
https://observablehq.com/@yourusername/clustering-tfjs-interactive-visualization
```

## Option 2: Fork and Modify

Alternatively, you can:

1. Search Observable for any existing clustering examples
2. Fork them to your account
3. Modify to use clustering-tfjs
4. Publish your version

## Option 3: Create a Collection

Once you have all notebooks:

1. Go to your Observable profile
2. Click "New collection"
3. Name it "clustering-tfjs Examples"
4. Add all four notebooks
5. Make the collection public

## Updating the README

Once you have the public URLs, update the main README.md:

```markdown
### Live Demos

Try these interactive examples directly in your browser:

- [**Interactive Clustering Visualization**](https://observablehq.com/@yourusername/clustering-tfjs-interactive-visualization) - Explore all algorithms with different datasets
- [**K-Means Animation**](https://observablehq.com/@yourusername/clustering-tfjs-kmeans-animation) - Watch K-Means algorithm in action
- [**Algorithm Comparison**](https://observablehq.com/@yourusername/clustering-tfjs-algorithm-comparison) - Compare performance across algorithms
- [**Quick Start Template**](https://observablehq.com/@yourusername/clustering-tfjs-quick-start) - Simple template to get started
```

## Tips for Public Notebooks

1. **Add a header cell** with markdown explaining what the notebook does
2. **Include links** back to the GitHub repo and npm package
3. **Add comments** in code cells to explain complex logic
4. **Use descriptive variable names** for clarity
5. **Include error handling** for edge cases
6. **Test on mobile** - Observable notebooks work on phones too!

## Promoting Your Notebooks

1. **Tweet** with #ObservableHQ and #MachineLearning hashtags
2. **Post** in Observable's community forum
3. **Add** to the clustering-tfjs GitHub README
4. **Submit** to Observable's trending notebooks
5. **Share** in relevant ML/data science communities

Would you like me to create the complete code for any of these notebooks?