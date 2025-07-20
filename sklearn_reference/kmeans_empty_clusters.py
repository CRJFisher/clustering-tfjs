"""
Analysis of sklearn's k-means empty cluster handling strategy.

From sklearn source code inspection (sklearn/cluster/_kmeans.py):

1. In the _kmeans_single_lloyd function, when an empty cluster is detected:
   ```python
   # Handle empty clusters
   n_samples_in_cluster = np.bincount(labels, minlength=n_clusters)
   empty_clusters = np.where(n_samples_in_cluster == 0)[0]
   
   if len(empty_clusters):
       # Find points with largest distance to any center
       distances = euclidean_distances(X, centers)
       distances_to_nearest = distances.min(axis=1)
       
       # Find farthest points
       farthest_idx = np.argpartition(distances_to_nearest, -len(empty_clusters))[-len(empty_clusters):]
       
       # Assign farthest points as new centers
       for i, cluster_id in enumerate(empty_clusters):
           centers[cluster_id] = X[farthest_idx[i]]
   ```

2. Key strategy:
   - Find all empty clusters
   - For each empty cluster, find the point that is farthest from its nearest center
   - Use that point as the new center for the empty cluster
   - This ensures maximum separation and helps avoid future empty clusters

3. Tie-breaking:
   - When multiple points have the same distance, np.argpartition provides
     deterministic ordering based on the array order
   - The random_state affects initial centers but not empty cluster handling

4. Important notes:
   - This happens AFTER label assignment in each iteration
   - Centers are updated in-place
   - The strategy is greedy - assigns farthest points one by one
   - No randomness involved in the reseeding itself
"""

print(__doc__)

# Let's create a test case that demonstrates empty clusters
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Create a dataset that is likely to produce empty clusters
np.random.seed(42)
X = np.vstack([
    np.random.randn(30, 2) * 0.5,  # Tight cluster at origin
    np.random.randn(5, 2) * 0.5 + [5, 5],  # Small cluster far away
    np.random.randn(5, 2) * 0.5 + [5, -5],  # Small cluster far away
])

print("Dataset shape:", X.shape)
print("3 distinct groups, but asking for 5 clusters\n")

# Run k-means with more clusters than natural groups
km = KMeans(n_clusters=5, n_init=1, init='random', random_state=42)
km.fit(X)

print("Final cluster sizes:", np.bincount(km.labels_))
print("Number of iterations:", km.n_iter_)

# Check if empty clusters occurred during fitting
# We can't directly observe this, but we can check the final state
unique_labels = np.unique(km.labels_)
if len(unique_labels) < 5:
    print(f"\nWARNING: Only {len(unique_labels)} clusters used out of 5!")
else:
    print("\nAll 5 clusters have points assigned")

# Visualize
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=km.labels_, cmap='tab10', alpha=0.6)
plt.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1], 
            marker='x', s=200, linewidths=3, color='black')
plt.title('K-means with potential empty clusters')
plt.savefig('kmeans_empty_clusters.png')
print("\nSaved visualization to kmeans_empty_clusters.png")