# Observable Examples for clustering-tfjs

This folder contains examples designed for Observable notebooks - the best platform for data visualization and machine learning demos.

## Why Observable?

- **Purpose-built for data visualization** - Created by the D3.js team
- **Excellent TensorFlow.js support** - Native integration with ML libraries
- **Reactive notebooks** - Live updating visualizations as parameters change
- **Free public notebooks** - Share your clustering demos with anyone

## Live Examples

### Quick Start Observable Notebook

Copy this code into a new Observable notebook at [observablehq.com](https://observablehq.com):

```javascript
// Cell 1: Load libraries
tf = require("https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.20.0/dist/tf.min.js")
ClusteringTFJS = require("https://unpkg.com/clustering-tfjs@0.3.1/dist/clustering.browser.js")

// Cell 2: Initialize clustering
clustering = {
  const { Clustering } = ClusteringTFJS;
  await Clustering.init({ backend: 'cpu' });
  return Clustering;
}

// Cell 3: Generate sample data
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

// Cell 4: Run K-Means clustering
clusters = {
  const kmeans = new clustering.KMeans({ nClusters: 3 });
  const labels = await kmeans.fitPredict(data);
  return data.map((point, i) => ({
    x: point[0],
    y: point[1],
    cluster: labels[i]
  }));
}

// Cell 5: Visualize results
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
  height: 400
})
```

### Coming Soon

Full interactive notebooks with:
- Parameter controls and animations
- Multiple algorithms comparison
- Performance benchmarks
- Real-world datasets

## How to Use These Examples

### Option 1: Fork on Observable
1. Click any link above
2. Sign in to Observable (free)
3. Click "Fork" to create your own copy
4. Modify and experiment!

### Option 2: Create New Notebook
1. Go to [observablehq.com](https://observablehq.com)
2. Create new notebook
3. Import clustering-tfjs:
```javascript
tf = require("https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.17.0/dist/tf.min.js")
ClusteringTFJS = require("https://unpkg.com/clustering-tfjs@latest/dist/clustering.browser.js")
```

### Option 3: Use Local HTML Files
The HTML files in this directory can also be:
- Opened directly in a browser
- Used as reference for Observable cells
- Adapted for other platforms (CodePen, JSFiddle)

## Example Observable Cell

```javascript
// Load libraries
tf = require("https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.17.0/dist/tf.min.js")
ClusteringTFJS = require("https://unpkg.com/clustering-tfjs@latest/dist/clustering.browser.js")

// Initialize
{
  const { Clustering } = ClusteringTFJS;
  await Clustering.init({ backend: 'cpu' });
  return Clustering;
}
```

```javascript
// Generate data
data = {
  const points = [];
  for (let i = 0; i < 3; i++) {
    const cx = Math.random() * 10;
    const cy = Math.random() * 10;
    for (let j = 0; j < 50; j++) {
      points.push({
        x: cx + (Math.random() - 0.5) * 2,
        y: cy + (Math.random() - 0.5) * 2,
        cluster: i
      });
    }
  }
  return points;
}
```

```javascript
// Run clustering
clusters = {
  const kmeans = new Clustering.KMeans({ nClusters: 3 });
  const dataArray = data.map(d => [d.x, d.y]);
  await kmeans.fit(dataArray);
  const labels = await kmeans.predict(dataArray);
  
  return data.map((d, i) => ({
    ...d,
    predicted: labels[i]
  }));
}
```

```javascript
// Visualize with D3
Plot.plot({
  marks: [
    Plot.dot(clusters, {
      x: "x",
      y: "y", 
      fill: "predicted",
      r: 5
    })
  ],
  color: {
    type: "categorical",
    scheme: "tableau10"
  }
})
```

## Converting HTML to Observable

To convert the HTML examples to Observable notebooks:

1. **Break into cells** - Each major section becomes a cell
2. **Use reactive programming** - Variables automatically update
3. **Leverage Observable Plot** - Modern declarative visualization
4. **Add sliders/inputs** - Use Observable's built-in inputs

Example conversion:
```javascript
// HTML version
const nClusters = document.getElementById('nClusters').value;

// Observable version
viewof nClusters = Inputs.range([2, 10], {step: 1, value: 3, label: "Number of clusters"})
```

## Resources

- [Observable Documentation](https://observablehq.com/@observablehq/documentation)
- [TensorFlow.js on Observable](https://observablehq.com/@tensorflow/tfjs-quick-start)
- [D3.js Gallery](https://observablehq.com/@d3/gallery)
- [Observable Plot](https://observablehq.com/plot/)