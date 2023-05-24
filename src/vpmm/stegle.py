import jax
import jax.numpy as jnp
from tensorflow_probability.substrates import jax as tfp

tfd = tfp.distributions


def stegle_vec(vec, sigma2, mat_lhs, mat_rhs) -> float:
    """
    calculates `vec' (sigma2 I + mat_lhs (x) mat_rhs)^{-1} vec` using stegle et al.

    mat_lhs, and mat_rhs can either be a matrix or a tuple with the result of
    the EVD using jnp.linalg.eigh(mat_lhs).

    vec must be a vector

    """

    if type(mat_lhs) is tuple:
        eig_val_lhs, eig_vec_lhs = mat_lhs
    else:
        eig_val_lhs, eig_vec_lhs = jnp.linalg.eigh(mat_lhs)

    if type(mat_rhs) is tuple:
        eig_val_rhs, eig_vec_rhs = mat_rhs
    else:
        eig_val_rhs, eig_vec_rhs = jnp.linalg.eigh(mat_rhs)

    size_lhs = eig_vec_lhs.shape[0]
    size_rhs = eig_vec_rhs.shape[0]

    vec_reshaped = vec.reshape(size_lhs, size_rhs).T
    vec_rotated = eig_vec_rhs.T @ vec_reshaped @ eig_vec_lhs
    vec_rotated = jnp.hstack(vec_rotated.T)

    weights = jnp.kron(eig_val_lhs, eig_val_rhs) + sigma2

    return jnp.sum(vec_rotated**2 / weights)


def stegle_vec2(v, w, sigma2, mat_lhs, mat_rhs) -> float:
    """
    calculates `v' (sigma2 I + mat_lhs (x) mat_rhs)^{-1} w` using stegle et al.

    mat_lhs, and mat_rhs can either be a matrix or a tuple with the result of
    the EVD using jnp.linalg.eigh(mat_lhs).

    v, w must be vectors

    """

    if type(mat_lhs) is tuple:
        eig_val_lhs, eig_vec_lhs = mat_lhs
    else:
        eig_val_lhs, eig_vec_lhs = jnp.linalg.eigh(mat_lhs)

    if type(mat_rhs) is tuple:
        eig_val_rhs, eig_vec_rhs = mat_rhs
    else:
        eig_val_rhs, eig_vec_rhs = jnp.linalg.eigh(mat_rhs)

    size_lhs = eig_vec_lhs.shape[0]
    size_rhs = eig_vec_rhs.shape[0]

    weights = jnp.sqrt(jnp.kron(eig_val_lhs, eig_val_rhs) + sigma2)

    v_reshaped = v.reshape(size_lhs, size_rhs).T
    v_rotated = eig_vec_rhs.T @ v_reshaped @ eig_vec_lhs
    v_rotated = jnp.hstack(v_rotated.T)
    v_weighted = v_rotated / weights

    w_reshaped = w.reshape(size_lhs, size_rhs).T
    w_rotated = eig_vec_rhs.T @ w_reshaped @ eig_vec_lhs
    w_rotated = jnp.hstack(w_rotated.T)
    w_weighted = w_rotated / weights

    return jnp.sum(v_weighted * w_weighted)


def stegle_mat(v, w, sigma2, mat_lhs, mat_rhs) -> float:
    """
    calculates `v' (sigma2 I + mat_lhs (x) mat_rhs)^{-1} w` using stegle et al.

    mat_lhs, and mat_rhs can either be a matrix or a tuple with the result of
    the EVD using jnp.linalg.eigh(mat_lhs).

    v, w can be matrices. the returned value is squeezed to fit the naive
    expression.

    """

    # convert v,w to matrices
    if v.ndim == 1:
        v = v.reshape(-1, 1)
    if w.ndim == 1:
        w = w.reshape(-1, 1)

    if type(mat_lhs) is tuple:
        eig_val_lhs, eig_vec_lhs = mat_lhs
    else:
        eig_val_lhs, eig_vec_lhs = jnp.linalg.eigh(mat_lhs)

    if type(mat_rhs) is tuple:
        eig_val_rhs, eig_vec_rhs = mat_rhs
    else:
        eig_val_rhs, eig_vec_rhs = jnp.linalg.eigh(mat_rhs)

    size_lhs = eig_vec_lhs.shape[0]
    size_rhs = eig_vec_rhs.shape[0]

    weights = jnp.sqrt(jnp.kron(eig_val_lhs, eig_val_rhs) + sigma2)

    # calcuates the result element-wise by considering one-row of v.T
    # and one column of w
    def map_v(i):
        vv = v[:, i]

        v_reshaped = vv.reshape(size_lhs, size_rhs).T
        v_rotated = eig_vec_rhs.T @ v_reshaped @ eig_vec_lhs
        v_rotated = jnp.hstack(v_rotated.T)
        v_weighted = v_rotated / weights
        return v_weighted

    v_weighted = jax.vmap(map_v)(jnp.arange(v.shape[1]))

    def map_w(j):
        ww = w[:, j]
        w_reshaped = ww.reshape(size_lhs, size_rhs).T
        w_rotated = eig_vec_rhs.T @ w_reshaped @ eig_vec_lhs
        w_rotated = jnp.hstack(w_rotated.T)
        w_weighted = w_rotated / weights
        return w_weighted

    w_weighted = jax.vmap(map_w)(jnp.arange(w.shape[1]))

    return (v_weighted @ w_weighted.T).squeeze()


def stegle_log_density(y, loc, sigma2, mat_lhs, mat_rhs, scale=1.0) -> float:
    """
    evaluates the log denisty for a mutlivariate normal distribution with covariance

    `scale**2 (sigma2 I + mat_lhs (x) mat_rhs)`

    and location `loc`
    """

    eig_cov_val_lhs, eig_cov_vec_lhs = jnp.linalg.eigh(mat_lhs)
    eig_cov_val_rhs, eig_cov_vec_rhs = jnp.linalg.eigh(mat_rhs)

    size_lhs = eig_cov_vec_lhs.shape[0]
    size_rhs = eig_cov_vec_rhs.shape[0]

    covariance_marginal = jnp.kron(eig_cov_val_lhs, eig_cov_val_rhs) + sigma2

    eps = y - loc

    new_eps = eps.reshape(size_lhs, size_rhs).T
    new_eps = eig_cov_vec_rhs.T @ new_eps @ eig_cov_vec_lhs
    new_eps = jnp.hstack(new_eps.T)

    lpdf_nd = tfd.MultivariateNormalDiag(
        loc=jnp.zeros_like(new_eps), scale_diag=jnp.sqrt(covariance_marginal) * scale
    )
    lpdf = lpdf_nd.log_prob(new_eps)
    return lpdf
