"""Tests some of the fast math functions in the library."""

from hr_selection_function.math import (
    vectorized_multivariate_normal,
    vectorized_1d_interpolation,
)
from scipy.stats import multivariate_normal
import numpy as np


def test_vectorized_multivariate_normal():
    rng = np.random.default_rng(42)
    _assert_multivariate_normal_results_identical(*_create_test_data(5, 2, rng))
    _assert_multivariate_normal_results_identical(*_create_test_data(5, 3, rng))
    _assert_multivariate_normal_results_identical(*_create_test_data(5, 5, rng))
    _assert_multivariate_normal_results_identical(*_create_test_data(1, 2, rng))
    _assert_multivariate_normal_results_identical(*_create_test_data(10, 3, rng))


def _make_valid_covariance_array(n_dims, rng):
    covariance = rng.uniform(size=(n_dims, n_dims))

    # Make it symmetric
    covariance = np.tril(covariance) + np.triu(covariance.T, 1)

    # Add offset to each term
    return covariance + np.diag(rng.uniform(10, 100, size=n_dims))


def _create_test_data(n_trials, n_dims, rng):
    means = rng.uniform(-10, 10, size=(n_trials, n_dims))
    covariances = np.asarray(
        [_make_valid_covariance_array(n_dims, rng) for i in range(n_trials)]
    )
    queries = rng.uniform(-50, 50, size=(n_trials, n_dims))
    return queries, means, covariances


def _assert_multivariate_normal_results_identical(queries, means, covariances):
    n_trials = means.shape[0]
    result_expected = np.zeros(n_trials)
    for i in range(n_trials):
        result_expected[i] = multivariate_normal(mean=means[i], cov=covariances[i]).pdf(
            queries[i]
        )

    result_vectorized = vectorized_multivariate_normal(queries, means, covariances)
    np.testing.assert_allclose(result_expected, result_vectorized)


def test_vectorized_1d_interpolation():
    n_trials = 10
    n_values = 50
    x_values = np.tile(np.linspace(0, 10, num=n_values), (n_trials, 1))

    offset = np.tile(np.linspace(0, 1, num=n_trials).reshape(-1, 1), (1, n_values))
    y_values = (x_values + offset) ** 2
    query_points = np.linspace(0, 10, num=n_trials)

    result_expected = np.zeros(n_trials)
    for i in range(n_trials):
        result_expected[i] = np.interp(query_points[i], x_values[i], y_values[i])

    result_vectorized = vectorized_1d_interpolation(query_points, x_values, y_values)

    np.testing.assert_allclose(result_expected, result_vectorized)
