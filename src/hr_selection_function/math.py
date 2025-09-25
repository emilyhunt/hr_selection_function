import numpy as np
from numba import njit


def vectorized_multivariate_normal(values, means, covariances):
    """This function exists here because scipy.stats.multivariate_normal can't be
    vectorized, and is hence really really slow =(
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
def vectorized_1d_interpolation(values, x, y):
    length = values.shape[0]
    result = np.zeros(length)
    for i in range(length):
        result[i] = np.interp(values[i], x[i], y[i])
    return result
