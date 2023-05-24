import jax.numpy as jnp
import numpy as np
import pytest
from tensorflow_probability.substrates import jax as tfp

from vpmm.stegle import stegle_log_density, stegle_mat, stegle_vec, stegle_vec2

tfd = tfp.distributions


def test_stegle_vec():
    mat_1 = jnp.array([[1.0, 0.4], [0.4, 1.0]])
    mat_2 = jnp.array([[1.0, 0.5], [0.5, 1.0]])
    sigma2 = 1.0
    v = jnp.array([0.0, 0.3, 0.2, 0.4])

    combined = jnp.kron(mat_1, mat_2)
    combined = combined.at[jnp.diag_indices_from(combined)].add(sigma2)
    combined_inv = jnp.linalg.inv(combined)

    result_naive = jnp.squeeze(v.T @ combined_inv @ v)
    result_stegle = stegle_vec(v, sigma2, mat_1, mat_2)

    assert pytest.approx(result_naive) == result_stegle


def test_stegle_vec_factors():
    mat_1 = jnp.array([[1.0, 0.5], [0.5, 1.0]])
    mat_2 = jnp.array([[1.0, 0.5], [0.5, 1.0]])
    sigma2 = 1.0
    v = jnp.array([0.0, 0.3, 0.2, 0.4])

    combined = jnp.kron(mat_1, mat_2)
    combined = combined.at[jnp.diag_indices_from(combined)].add(sigma2)
    combined_inv = jnp.linalg.inv(combined)

    result_naive = jnp.squeeze(v.T @ combined_inv @ v)
    result_stegle = stegle_vec(
        v, sigma2, jnp.linalg.eigh(mat_1), jnp.linalg.eigh(mat_2)
    )

    assert pytest.approx(result_naive) == result_stegle


def test_stegle_vec2():
    mat_1 = jnp.array([[1.0, 0.5], [0.5, 1.0]])
    mat_2 = jnp.array([[1.0, 0.6], [0.6, 1.0]])
    sigma2 = 1.0
    v = jnp.array([0.0, 0.3, 0.2, 0.4])
    w = jnp.array([0.1, 0.2, 0.1, 0.0])

    combined = jnp.kron(mat_1, mat_2)
    combined = combined.at[jnp.diag_indices_from(combined)].add(sigma2)
    combined_inv = jnp.linalg.inv(combined)

    result_naive = jnp.squeeze(v.T @ combined_inv @ w)
    result_stegle = stegle_vec2(v, w, sigma2, mat_1, mat_2)

    assert pytest.approx(result_naive) == result_stegle


def test_stegle_vec2_factor():
    mat_1 = jnp.array([[1.0, 0.5], [0.5, 1.0]])
    mat_2 = jnp.array([[1.0, 0.6], [0.6, 1.0]])
    sigma2 = 1.0
    v = jnp.array([0.0, 0.3, 0.2, 0.4])
    w = jnp.array([0.1, 0.2, 0.1, 0.0])

    combined = jnp.kron(mat_1, mat_2)
    combined = combined.at[jnp.diag_indices_from(combined)].add(sigma2)
    combined_inv = jnp.linalg.inv(combined)

    result_naive = jnp.squeeze(v.T @ combined_inv @ w)
    result_stegle = stegle_vec2(
        v, w, sigma2, jnp.linalg.eigh(mat_1), jnp.linalg.eigh(mat_2)
    )

    assert pytest.approx(result_naive) == result_stegle


def test_stegle_mat():
    mat_1 = jnp.array([[1.0, 0.5], [0.5, 1.0]])
    mat_2 = jnp.array([[1.0, 0.6], [0.6, 1.0]])
    sigma2 = 1.0
    v = jnp.array([[0.0, 0.3, 0.2, 0.4], [0.1, 0.3, 0.2, 0.5]]).T
    w = jnp.array([[0.1, 0.2, 0.1, 0.0], [0.1, 0.0, 0.0, 0.1]]).T

    combined = jnp.kron(mat_1, mat_2)
    combined = combined.at[jnp.diag_indices_from(combined)].add(sigma2)
    combined_inv = jnp.linalg.inv(combined)

    result_naive = v.T @ combined_inv @ w
    result_stegle = stegle_mat(v, w, sigma2, mat_1, mat_2)

    assert np.allclose(result_naive, result_stegle)


def test_stegle_mat_vec():
    mat_1 = jnp.array([[1.0, 0.5], [0.5, 1.0]])
    mat_2 = jnp.array([[1.0, 0.6], [0.6, 1.0]])
    sigma2 = 1.0
    v = jnp.array([[0.0, 0.3, 0.2, 0.4], [0.1, 0.3, 0.2, 0.5]]).T
    w = jnp.array([0.1, 0.2, 0.1, 0.0])

    combined = jnp.kron(mat_1, mat_2)
    combined = combined.at[jnp.diag_indices_from(combined)].add(sigma2)
    combined_inv = jnp.linalg.inv(combined)

    result_naive = v.T @ combined_inv @ w
    result_stegle = stegle_mat(v, w, sigma2, mat_1, mat_2)
    assert np.allclose(result_naive, result_stegle)


def test_stegle_mat_vec_vec():
    mat_1 = jnp.array([[1.0, 0.5], [0.5, 1.0]])
    mat_2 = jnp.array([[1.0, 0.6], [0.6, 1.0]])
    sigma2 = 1.0
    v = jnp.array([0.0, 0.3, 0.2, 0.4]).T
    w = jnp.array([0.1, 0.2, 0.1, 0.0])

    combined = jnp.kron(mat_1, mat_2)
    combined = combined.at[jnp.diag_indices_from(combined)].add(sigma2)
    combined_inv = jnp.linalg.inv(combined)

    result_naive = v.T @ combined_inv @ w
    result_stegle = stegle_mat(v, w, sigma2, mat_1, mat_2)
    assert np.allclose(result_naive, result_stegle)


def test_stegle_log_density():
    mat_1 = jnp.array([[1.0, 0.5], [0.5, 1.0]])
    mat_2 = jnp.array([[1.0, 0.6], [0.6, 1.0]])
    sigma2 = 0.9
    y = jnp.array([0.0, 0.3, 0.2, 0.4])
    loc = jnp.array([0.1, 0.2, 0.1, 0.0])

    combined = jnp.kron(mat_1, mat_2)
    combined = combined.at[jnp.diag_indices_from(combined)].add(sigma2)
    dist_naive = tfd.MultivariateNormalFullCovariance(loc, combined)

    result_naive = dist_naive.log_prob(y)
    result_stegle = stegle_log_density(
        y,
        loc,
        sigma2,
        mat_1,
        mat_2,
    )
    assert np.allclose(result_naive, result_stegle)


def test_stegle_log_density_with_scale():
    mat_1 = jnp.array([[1.0, 0.5], [0.5, 1.0]])
    mat_2 = jnp.array([[1.0, 0.6], [0.6, 1.0]])
    sigma2 = 0.9
    scale = 1.2
    y = jnp.array([0.0, 0.3, 0.2, 0.4])
    loc = jnp.array([0.1, 0.2, 0.1, 0.0])

    combined = jnp.kron(mat_1, mat_2)
    combined = combined.at[jnp.diag_indices_from(combined)].add(sigma2)
    dist_naive = tfd.MultivariateNormalFullCovariance(loc, combined * scale**2)

    result_naive = dist_naive.log_prob(y)
    result_stegle = stegle_log_density(y, loc, sigma2, mat_1, mat_2, scale)
    assert np.allclose(result_naive, result_stegle)
