# From sklearn/cluster/_spectral.py
# Key parts of SpectralClustering implementation

class SpectralClustering(BaseEstimator, ClusterMixin):
    """Apply clustering to a projection of the normalized Laplacian."""
    
    def __init__(
        self,
        n_clusters=8,
        *,
        eigen_solver=None,
        n_components=None,
        random_state=None,
        n_init=10,
        gamma=1.0,
        affinity="rbf",
        n_neighbors=10,
        eigen_tol="auto",
        assign_labels="kmeans",
        degree=3,
        coef0=1,
        kernel_params=None,
        n_jobs=None,
        verbose=False,
    ):
        self.n_clusters = n_clusters
        self.eigen_solver = eigen_solver
        self.n_components = n_components
        self.random_state = random_state
        self.n_init = n_init
        self.gamma = gamma
        self.affinity = affinity
        self.n_neighbors = n_neighbors
        self.eigen_tol = eigen_tol
        self.assign_labels = assign_labels
        self.degree = degree
        self.coef0 = coef0
        self.kernel_params = kernel_params
        self.n_jobs = n_jobs
        self.verbose = verbose

    def fit(self, X, y=None):
        """Perform spectral clustering from features, or affinity matrix."""
        # ... validation code ...
        
        # Key part: affinity matrix construction
        if self.affinity == "rbf":
            self.gamma_ = self.gamma if self.gamma is not None else 1.0 / X.shape[1]
            self.affinity_matrix_ = rbf_kernel(X, gamma=self.gamma_)
        elif self.affinity == "nearest_neighbors":
            connectivity = kneighbors_graph(
                X, n_neighbors=self.n_neighbors, include_self=True, n_jobs=self.n_jobs
            )
            self.affinity_matrix_ = 0.5 * (connectivity + connectivity.T)
        # ... other affinity types ...
        
        # Apply spectral clustering
        self.labels_ = spectral_clustering(
            self.affinity_matrix_,
            n_clusters=self.n_clusters,
            n_components=self.n_components,
            eigen_solver=self.eigen_solver,
            random_state=random_state,
            n_init=self.n_init,
            eigen_tol=self.eigen_tol,
            assign_labels=self.assign_labels,
            verbose=self.verbose,
        )
        return self


def spectral_clustering(
    affinity,
    *,
    n_clusters=8,
    n_components=None,
    eigen_solver=None,
    random_state=None,
    n_init=10,
    eigen_tol="auto",
    assign_labels="kmeans",
    verbose=False,
):
    """Apply clustering to a projection of the normalized Laplacian."""
    
    # The key steps:
    # 1. Compute normalized Laplacian
    # 2. Compute first k eigenvectors 
    # 3. Apply k-means to the embedding
    
    # Important: gamma default calculation
    # If gamma is None, it defaults to 1.0 / n_features
    # This is crucial for RBF kernel!