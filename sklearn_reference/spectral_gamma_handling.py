# Investigation of sklearn's gamma handling in SpectralClustering

# From sklearn source code analysis:

# 1. In SpectralClustering.__init__:
#    self.gamma = gamma  # Stores the user-provided gamma

# 2. In SpectralClustering.fit:
#    When affinity='rbf', it calls:
#    self.affinity_matrix_ = rbf_kernel(X, gamma=self.gamma)

# 3. In rbf_kernel:
#    if gamma is None:
#        gamma = 1.0 / X.shape[1]
#    
#    K = euclidean_distances(X, Y, squared=True)
#    K *= -gamma
#    np.exp(K, K)  # exponentiate K in-place
#    return K

# So sklearn does NOT auto-scale gamma beyond the default 1/n_features when None

# However, there might be an issue with how the fixture was generated.
# Looking at generate_spectral.py line 48:
# {"n_clusters": 2, "affinity": "rbf", "gamma": 1.0},

# This explicitly sets gamma=1.0, which overrides sklearn's default.
# But the actual labels in the fixture might have been generated with a different gamma!

# Possible explanations:
# 1. The fixture generation script was changed after generating the fixtures
# 2. There's a bug in the fixture generation where gamma wasn't properly passed
# 3. The fixtures were generated with an older version of sklearn that handled gamma differently

# Most likely: The fixtures were generated with sklearn's default gamma (1/n_features = 0.5)
# but the fixture JSON incorrectly records gamma=1.0