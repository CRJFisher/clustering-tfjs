# Investigating sklearn's gamma interpretation

# From sklearn documentation and source:
# The RBF kernel is defined as:
# K(x, y) = exp(-gamma ||x-y||^2)

# However, there's a common alternative formulation:
# K(x, y) = exp(-||x-y||^2 / (2 * sigma^2))
# where gamma = 1 / (2 * sigma^2)

# Some libraries use:
# K(x, y) = exp(-||x-y||^2 / (2 * gamma))
# which would make gamma = sigma^2

# Let's check sklearn's exact implementation
# From sklearn/metrics/pairwise.py:
"""
def rbf_kernel(X, Y=None, gamma=None):
    '''
    Compute the rbf (gaussian) kernel between X and Y.

    K(x, y) = exp(-gamma ||x-y||^2)

    for each pair of rows x in X and y in Y.
    '''
    X, Y = check_pairwise_arrays(X, Y)
    if gamma is None:
        gamma = 1.0 / X.shape[1]

    K = euclidean_distances(X, Y, squared=True)
    K *= -gamma
    np.exp(K, K)  # exponentiate K in-place
    return K
"""

# So sklearn uses: K(x, y) = exp(-gamma ||x-y||^2)
# This matches our implementation!

# The issue might be in how the gamma value is chosen in the fixture
# Let's check if there's a scaling factor we're missing