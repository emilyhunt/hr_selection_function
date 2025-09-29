import numpy as np
from numba import njit


def vectorized_multivariate_normal(
    values: np.ndarray, means: np.ndarray, covariances: np.ndarray
) -> np.ndarray:
    """Calculate the PDF of n_queries different multivariate normal distributions at
    n_queries different points in a fast and vectorized way. Significantly faster than
    scipy.stats.multivariate_normal.pdf() when n_queries is large.

    Parameters
    ----------
    values : np.ndarray
        Values to query the PDF at. Must have shape (n_queries, n_dims).
    means : np.ndarray
        Means of the distributions. Must have shape (n_queries, n_dims).
    covariances : np.ndarray
        Covariances of the distributions. Must have shape (n_queries, n_dims, n_dims).

    Returns
    -------
    np.ndarray
        Array of shape (n_queries,) giving the PDF value of each distribution.
    """
    k = means.shape[1]

    # Calculate products from covariances
    means = means.reshape(-1, k, 1)
    values = values.reshape(-1, k, 1)
    covariances = covariances.reshape(-1, k, k)
    determinants = np.linalg.det(covariances)
    inverses = np.linalg.inv(covariances)

    diff = values - means

    constant = (2 * np.pi) ** (-k / 2) * determinants ** (-0.5)
    exponent = -0.5 * (diff.reshape(-1, 1, k) @ inverses @ diff).flatten()

    return constant.flatten() * np.exp(exponent.flatten())


@njit
def vectorized_1d_interpolation(
    values: np.ndarray, x: np.ndarray, y: np.ndarray
) -> np.ndarray:
    """Vectorized, numba version of np.interp.

    Parameters
    ----------
    values : ArrayLike
        Values to query, of shape (n_queries,).
    x : ArrayLike
        x values of data, of shape (n_queries, n_data_points).
    y : ArrayLike
        y values of data, of shape (n_queries, n_data_points).

    Returns
    -------
    np.ndarray
        Interpolated values array of shape (n_queries,).
    """
    length = values.shape[0]
    result = np.zeros(length)
    for i in range(length):
        result[i] = np.interp(values[i], x[i], y[i])
    return result
