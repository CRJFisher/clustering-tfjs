# From sklearn/metrics/pairwise.py
# RBF kernel implementation from scikit-learn

def rbf_kernel(X, Y=None, gamma=None):
    """
    Compute the rbf (gaussian) kernel between X and Y.

    K(x, y) = exp(-gamma ||x-y||^2)

    for each pair of rows x in X and y in Y.

    Parameters
    ----------
    X : ndarray of shape (n_samples_X, n_features)
        A feature array.

    Y : ndarray of shape (n_samples_Y, n_features), default=None
        An optional second feature array. If None, uses Y=X.

    gamma : float, default=None
        If None, defaults to 1.0 / n_features

    Returns
    -------
    kernel_matrix : ndarray of shape (n_samples_X, n_samples_Y)
        The RBF kernel.
    """
    X, Y = check_pairwise_arrays(X, Y)
    if gamma is None:
        gamma = 1.0 / X.shape[1]

    K = euclidean_distances(X, Y, squared=True)
    K *= -gamma
    np.exp(K, K)  # exponentiate K in-place
    return K


def euclidean_distances(X, Y=None, *, Y_norm_squared=None, squared=False, X_norm_squared=None):
    """
    Compute the distance matrix between each pair from a vector array X and Y.

    For efficiency reasons, the euclidean distance between a pair of row
    vector x and y is computed as::

        dist(x, y) = sqrt(dot(x, x) - 2 * dot(x, y) + dot(y, y))

    This formulation has two advantages over other ways of computing distances.
    First, it is computationally efficient when dealing with sparse data.
    Second, if one argument varies but the other remains unchanged, then
    `dot(x, x)` and/or `dot(y, y)` can be pre-computed.

    However, this is not the most precise way of doing this computation,
    because this equation potentially suffers from "catastrophic cancellation".
    Also, the distance matrix returned by this function may not be exactly
    symmetric as required by, e.g., scipy.spatial.distance functions.
    """
    # ... implementation details ...
    
    # Key point: When Y is None, Y = X
    # The squared euclidean distance is computed as:
    # ||x - y||^2 = ||x||^2 + ||y||^2 - 2 * x.y
    
    # For RBF in spectral clustering, the similarity matrix is:
    # S[i,j] = exp(-gamma * ||X[i] - X[j]||^2)