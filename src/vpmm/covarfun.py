import jax.numpy as jnp


def exponential_covariance(h, kappa, sigma2):
    """
    exponential covariance function. kappa = 3/range.
    """
    cov = sigma2 * jnp.exp(-kappa * h)

    return cov
