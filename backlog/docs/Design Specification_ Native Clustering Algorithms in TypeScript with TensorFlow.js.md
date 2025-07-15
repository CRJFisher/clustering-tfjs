# Design Specification: Native Clustering Algorithms in TypeScript with TensorFlow.js

## 1. Overview

This document provides a technical design specification for implementing `AgglomerativeClustering`, `SpectralClustering`, and the validation metrics `calinski_harabasz_score`, `davies_bouldin_score`, and `silhouette_score` in TypeScript.  
The primary goal is to create a high-performance, dependency-free library suitable for a Node.js environment, such as a VS Code extension. All numerical computations will be implemented using the TensorFlow.js library to leverage its performance optimizations, hardware acceleration capabilities, and comprehensive linear algebra API.[1]

## 2. Core Dependencies and Setup

This implementation will be built on a modern TypeScript and Node.js stack.

- **Language:** TypeScript 3
- **Core Numerical Library:** TensorFlow.js. For a VS Code extension environment, the native C++ bindings are essential for performance.[4]
  - `@tensorflow/tfjs`: The core library.
  - `@tensorflow/tfjs-node`: Provides native TensorFlow execution on Node.js, which is significantly faster than the pure JS backend.[5]
- **Project Configuration:** A standard `tsconfig.json` will be used to configure the TypeScript compiler.[3]

```json
// tsconfig.json
{
  "compilerOptions": {
    "target": "es2020",
    "module": "commonjs",
    "strict": true,
    "esModuleInterop": true,
    "skipLibCheck": true
  }
}
```

## 3. API Design and Conventions

The public API will be designed to mirror the familiar structure of scikit-learn, adapted for TypeScript/JavaScript conventions.[6]

- **Naming:** camelCase will be used for all functions, methods, and properties (e.g., `fitPredict`, `nClusters`).
- **Estimator Instantiation:** Class constructors will accept a single configuration object for parameters, which is a common JavaScript pattern for handling numerous optional arguments.[6]
- **Data Format:** Input data `X` will be accepted as `number`. Internally, all data will be converted to and processed as `tf.Tensor` objects for performance.
- **Asynchronous Operations:** All fitting and prediction methods will be async and return a Promise, as underlying TensorFlow.js operations are often asynchronous.
- **Memory Management:** All internal tensor computations will be wrapped in `tf.tidy()` blocks to prevent memory leaks by automatically disposing of intermediate tensors.

## 4. Component Design: Algorithms

### 4.1 AgglomerativeClustering

This class will implement a bottom-up hierarchical clustering algorithm.[7]

#### Public API

```typescript
interface AgglomerativeClusteringParams {
  nClusters?: number;
  linkage?: "ward" | "complete" | "average" | "single";
  metric?: "euclidean" | "manhattan" | "cosine";
}

class AgglomerativeClustering {
  constructor(params: AgglomerativeClusteringParams);

  labels_: tf.Tensor1D;
  children_: tf.Tensor2D;
  nLeaves_: number;

  async fit(X: number): Promise<this>;
  async fitPredict(X: number): Promise<number>;
}
```

#### Implementation Blueprint

1. **Initialization:** In the constructor, store parameters. Initialize n clusters, where n is the number of samples.
2. **Pairwise Distance Matrix:**
   - Convert input X to a `tf.Tensor2D`.
   - Compute the initial n x n pairwise distance matrix based on the specified metric. This can be implemented using broadcasted tensor operations (e.g., `tf.sub`, `tf.squaredDifference`, `tf.norm`).
3. **Iterative Merging:** Loop n - 1 times or until nClusters is reached.
   - **Find Closest Pair:** Find the minimum value in the distance matrix to identify the two clusters to merge.
   - **Linkage Criteria:** Implement the logic for updating the distance matrix after a merge. This is the core of the algorithm.[9]
     - **single:** Minimum distance between points in the two clusters.
     - **complete:** Maximum distance between points in the two clusters.
     - **average:** Average distance between all pairs of points.
     - **ward:** Minimizes the increase in total within-cluster variance. This requires calculating cluster centroids and variances using tensor operations.
   - **Update State:** Store the merge information in the `children_` tensor and update the distance matrix for the next iteration.
4. **Label Assignment:** After the loop, traverse the `children_` hierarchy to assign final cluster labels to each sample.

### 4.2 SpectralClustering

This class will implement clustering on a spectral embedding of the data, which is effective for non-convex cluster shapes.[11]

#### Public API

```typescript
interface SpectralClusteringParams {
  nClusters?: number;
  affinity?: "rbf" | "nearest_neighbors";
  gamma?: number; // For 'rbf' kernel
  nNeighbors?: number; // For 'nearest_neighbors'
}

class SpectralClustering {
  constructor(params: SpectralClusteringParams);

  labels_: tf.Tensor1D;
  affinityMatrix_: tf.Tensor2D;

  async fit(X: number): Promise<this>;
  async fitPredict(X: number): Promise<number>;
}
```

#### Implementation Blueprint

1. **Affinity Matrix Construction:** The first step is to build a similarity graph represented by an affinity matrix A.[13]
   - **rbf (Gaussian Kernel):** Compute pairwise squared Euclidean distances and apply the formula \(A\_{ij} = \exp(-\gamma \|x_i - x_j\|^2)\) using `tf.exp` and tensor arithmetic.
   - **nearest_neighbors:** For each point, find its k nearest neighbors using distance calculations and `tf.topk`. Construct a sparse affinity matrix where \(A\_{ij} = 1\) if i is a neighbor of j or vice-versa.
2. **Graph Laplacian Computation:** From the affinity matrix A, compute the symmetrically normalized Laplacian, which is preferred for its mathematical properties.[14]
   - Compute the Degree Matrix D by summing the rows of A (`tf.sum`).
   - Calculate \(L\_{sym} = I - D^{-1/2} A D^{-1/2}\) using `tf.matMul`, `tf.sqrt`, and `tf.reciprocal`.
3. **Eigendecomposition:** This is the core "spectral" step.[16]
   - Use `tf.linalg.eig(laplacian)` to compute the eigenvalues and eigenvectors of the Laplacian matrix. This function is critical and is a primary reason for choosing TensorFlow.js as the backend.
4. **Spectral Embedding:**
   - Sort the eigenvalues in ascending order.
   - Select the k eigenvectors corresponding to the k smallest eigenvalues (where k is nClusters).
   - Form a new n x k matrix U where the columns are these selected eigenvectors. This is the spectral embedding of the data into a lower-dimensional space.
5. **Final Clustering:**
   - Treat each row of the embedding matrix U as a new data point.
   - Apply a standard K-Means clustering algorithm to these n new points to get the final cluster labels. This requires a helper implementation of K-Means.

## 5. Component Design: Validation Metrics

These will be implemented as standalone, asynchronous functions that operate on tensors.

### 5.1 CalinskiHarabaszScore

Measures the ratio of between-cluster to within-cluster dispersion. Higher is better.[18]

- **Signature:** `async function calinskiHarabaszScore(X: number, labels: number): Promise<number>`
- **Formula:** \(CH = \frac{WSS/(N-k)}{BSS/(k-1)}\) [19]
- **Implementation:**
  1. Convert inputs X and labels to tensors.
  2. Calculate the global centroid c of all points.
  3. For each cluster, calculate its centroid c_i and the number of points n_i. This can be done efficiently using `tf.gather` and segmented reductions.
  4. Calculate **WSS** (Within-Cluster Sum of Squares): For each cluster, sum the squared Euclidean distances from each point to its cluster centroid. Sum these values across all clusters.
  5. Calculate **BSS** (Between-Cluster Sum of Squares): For each cluster, compute the squared Euclidean distance from its centroid c_i to the global centroid c, weighted by the cluster size n_i. Sum these values.
  6. Apply the final formula using the calculated WSS, BSS, N (total points), and k (number of clusters).

### 5.2 DaviesBouldinScore

Measures the average similarity between each cluster and its most similar one. Lower is better (minimum 0).[21]

- **Signature:** `async function daviesBouldinScore(X: number, labels: number): Promise<number>`
- **Formula:** \(DB = \frac{1}{k} \sum*{i=1}^k \max*{j \neq i} \frac{s_i + s_j}{d(c_i, c_j)}\) [23]
- **Implementation:**
  1. Convert inputs to tensors.
  2. Calculate the centroid c_i for each cluster i.
  3. Calculate the intra-cluster dispersion s_i for each cluster (average distance from points to their centroid).
  4. Calculate the inter-cluster distance d(c_i, c_j) for all pairs of centroids.
  5. For each cluster i, compute the similarity ratio with all other clusters j and find the maximum value (R_i).
  6. The final score is the average of these maximum values over all clusters.

### 5.3 SilhouetteScore

Measures how similar a point is to its own cluster compared to other clusters. Ranges from -1 to +1, where higher is better.[25]

- **Signature:** `async function silhouetteScore(X: number, labels: number): Promise<number>`
- **Formula:** For a single sample i, \(s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}\). The final score is the mean of s(i) over all samples.[27]
- **Implementation:**
  1. Convert inputs to tensors.
  2. Compute the full n x n pairwise distance matrix for X.
  3. For each sample i:
     - **a(i) (Cohesion):** Calculate the mean distance from i to all other points in its own cluster. This can be done by creating a mask from the labels tensor to select the relevant distances.
     - **b(i) (Separation):** For every _other_ cluster, calculate the mean distance from i to all points in that cluster. b(i) is the minimum of these mean distances.
  4. Calculate s(i) for each sample using the formula.
  5. Compute the mean of all s(i) values using `tf.mean`.

## 6. Testing Strategy

A robust testing suite is critical to ensure correctness and numerical stability.

- **Unit Tests:** Each class method and metric function will be unit-tested in isolation using a framework like **Jest**. Tests will cover edge cases (e.g., empty input, single-cluster data).
- **Validation Tests:** The outputs of the TypeScript implementations will be compared against the reference scikit-learn Python implementations.
  1. Generate several synthetic datasets (e.g., using `make_blobs`).
  2. Run the Python scikit-learn functions on this data and save the results (labels, scores).
  3. Run the new TypeScript functions on the same data.
  4. Assert that the TypeScript outputs are numerically close to the Python outputs within a defined tolerance (e.g., 1e-6) to account for floating-point differences.

---

### Works cited

1. tensorflow/tfjs: A WebGL accelerated JavaScript library for training and deploying ML models. - GitHub, accessed on July 15, 2025, [https://github.com/tensorflow/tfjs](https://github.com/tensorflow/tfjs)
2. Introduction To TensorFlow.js - GeeksforGeeks, accessed on July 15, 2025, [https://www.geeksforgeeks.org/javascript/tensorflow-js/](https://www.geeksforgeeks.org/javascript/tensorflow-js/)
3. Practical Uses for TensorFlow.js with TypeScript Examples - Medium, accessed on July 15, 2025, [https://medium.com/@Jesse_Reese/practical-uses-for-tensorflow-js-with-typescript-examples-7dd01d5c8698](https://medium.com/@Jesse_Reese/practical-uses-for-tensorflow-js-with-typescript-examples-7dd01d5c8698)
4. scikitjs - NPM, accessed on July 15, 2025, [https://www.npmjs.com/package/scikitjs](https://www.npmjs.com/package/scikitjs)
5. tensorflow/tfjs-node - NPM, accessed on July 15, 2025, [https://www.npmjs.com/package/@tensorflow/tfjs-node](https://www.npmjs.com/package/@tensorflow/tfjs-node)
6. scikitjs-node - NPM, accessed on July 15, 2025, [https://www.npmjs.com/package/scikitjs-node](https://www.npmjs.com/package/scikitjs-node)
7. What is Agglomerative clustering ? - Educative.io, accessed on July 15, 2025, [https://www.educative.io/answers/what-is-agglomerative-clustering](https://www.educative.io/answers/what-is-agglomerative-clustering)
8. Implementing Agglomerative Clustering using Sklearn - GeeksforGeeks, accessed on July 15, 2025, [https://www.geeksforgeeks.org/machine-learning/implementing-agglomerative-clustering-using-sklearn/](https://www.geeksforgeeks.org/machine-learning/implementing-agglomerative-clustering-using-sklearn/)
9. 7 Steps to Master Agglomerative Clustering: A Statistical Approach - Number Analytics, accessed on July 15, 2025, [https://www.numberanalytics.com/blog/7-steps-to-master-agglomerative-clustering](https://www.numberanalytics.com/blog/7-steps-to-master-agglomerative-clustering)
10. Hierarchical Clustering: Agglomerative and Divisive Explained, accessed on July 15, 2025, [https://builtin.com/machine-learning/agglomerative-clustering](https://builtin.com/machine-learning/agglomerative-clustering)
11. SpectralClustering — scikit-learn 1.7.0 documentation, accessed on July 15, 2025, [https://scikit-learn.org/stable/modules/generated/sklearn.cluster.SpectralClustering.html](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.SpectralClustering.html)
12. Spectral Clustering in Machine Learning - GeeksforGeeks, accessed on July 15, 2025, [https://www.geeksforgeeks.org/machine-learning/ml-spectral-clustering/](https://www.geeksforgeeks.org/machine-learning/ml-spectral-clustering/)
13. A Tutorial on Spectral Clustering - CMU School of Computer Science, accessed on July 15, 2025, [https://www.cs.cmu.edu/~aarti/Class/10701/readings/Luxburg06_TR.pdf](https://www.cs.cmu.edu/~aarti/Class/10701/readings/Luxburg06_TR.pdf)
14. Mining-Massive-Datasets/Spectral clustering/Spectral Clustering.ipynb at master - GitHub, accessed on July 15, 2025, [https://github.com/PiotrTa/Mining-Massive-Datasets/blob/master/Spectral%20clustering/Spectral%20Clustering.ipynb](https://github.com/PiotrTa/Mining-Massive-Datasets/blob/master/Spectral%20clustering/Spectral%20Clustering.ipynb)
15. Getting Started with Spectral Clustering - Dr. Juan Camilo Orduz, accessed on July 15, 2025, [https://juanitorduz.github.io/spectral_clustering/](https://juanitorduz.github.io/spectral_clustering/)
16. Spectral Clustering — Machine Learning for Engineers - APMonitor, accessed on July 15, 2025, [https://apmonitor.com/pds/index.php/Main/SpectralClustering](https://apmonitor.com/pds/index.php/Main/SpectralClustering)
17. Spectral clustering - Wikipedia, accessed on July 15, 2025, [https://en.wikipedia.org/wiki/Spectral_clustering](https://en.wikipedia.org/wiki/Spectral_clustering)
18. Calinski–Harabasz index - Wikipedia, accessed on July 15, 2025, [https://en.wikipedia.org/wiki/Calinski%E2%80%93Harabasz_index](https://en.wikipedia.org/wiki/Calinski%E2%80%93Harabasz_index)
19. CalinskiHarabaszEvaluation - Calinski-Harabasz criterion clustering evaluation object - MATLAB - MathWorks, accessed on July 15, 2025, [https://www.mathworks.com/help/stats/clustering.evaluation.calinskiharabaszevaluation.html](https://www.mathworks.com/help/stats/clustering.evaluation.calinskiharabaszevaluation.html)
20. Uncovering the Optimal Number of Clusters: Part 1 “Introduction & Background” - Medium, accessed on July 15, 2025, [https://medium.com/@ryassminh/uncovering-the-optimal-number-of-clusters-part-1-introduction-background-79862df1d313](https://medium.com/@ryassminh/uncovering-the-optimal-number-of-clusters-part-1-introduction-background-79862df1d313)
21. Davies–Bouldin index - Wikipedia, accessed on July 15, 2025, [https://en.wikipedia.org/wiki/Davies%E2%80%93Bouldin_index](https://en.wikipedia.org/wiki/Davies%E2%80%93Bouldin_index)
22. Optimizing Machine Learning Clusters: Efficient Strategies with Davies-Bouldin Metrics, accessed on July 15, 2025, [https://www.numberanalytics.com/blog/optimizing-machine-learning-clusters-davies-bouldin-metrics](https://www.numberanalytics.com/blog/optimizing-machine-learning-clusters-davies-bouldin-metrics)
23. Mastering Clustering: A Guided Tour of the Davies-Bouldin Index - Number Analytics, accessed on July 15, 2025, [https://www.numberanalytics.com/blog/mastering-clustering-davies-bouldin-index](https://www.numberanalytics.com/blog/mastering-clustering-davies-bouldin-index)
24. Mastering the Davies-Bouldin Index for Clustering Model Validation | CodeSignal Learn, accessed on July 15, 2025, [https://codesignal.com/learn/courses/cluster-performance-unveiled/lessons/mastering-the-davies-bouldin-index-for-clustering-model-validation](https://codesignal.com/learn/courses/cluster-performance-unveiled/lessons/mastering-the-davies-bouldin-index-for-clustering-model-validation)
25. Silhouette (clustering) - Wikipedia, accessed on July 15, 2025, [https://en.wikipedia.org/wiki/Silhouette\_(clustering)](<https://en.wikipedia.org/wiki/Silhouette_(clustering)>)
26. Understanding Silhouette Score in Clustering | by FARSHAD K - Medium, accessed on July 15, 2025, [https://farshadabdulazeez.medium.com/understanding-silhouette-score-in-clustering-8aedc06ce9c4](https://farshadabdulazeez.medium.com/understanding-silhouette-score-in-clustering-8aedc06ce9c4)
27. What is Silhouette Score? - GeeksforGeeks, accessed on July 15, 2025, [https://www.geeksforgeeks.org/machine-learning/what-is-silhouette-score/](https://www.geeksforgeeks.org/machine-learning/what-is-silhouette-score/)
28. Silhouette Coefficient - KNIME Community Hub, accessed on July 15, 2025, [https://hub.knime.com/knime/extensions/org.knime.features.distmatrix/latest/org.knime.base.node.mine.cluster.eval.silhouette.SilhouetteCoefficientNodeFactory](https://hub.knime.com/knime/extensions/org.knime.features.distmatrix/latest/org.knime.base.node.mine.cluster.eval.silhouette.SilhouetteCoefficientNodeFactory)
