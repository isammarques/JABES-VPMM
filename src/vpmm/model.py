"""
Implements the model
"""

from dataclasses import dataclass
from functools import partial
from typing import Any, ClassVar, NamedTuple

import jax
import jax.numpy as jnp
import liesel.goose as gs
import scipy.spatial as spat
from liesel.goose.pytree import register_dataclass_as_pytree
from tensorflow_probability.substrates import jax as tfp

from .bmpriors import (
    KappaPriorNormal,
    KappaPriorNormal2,
    KappaPriorUniform,
    VarPriorHalfCauchy,
    VarPriorHalfNormal,
    VarPriorInvGamma,
)
from .conn import DataClassConnector
from .covarfun import exponential_covariance
from .quantgen_simple import quantgen
from .stegle import stegle_log_density, stegle_mat

tfd = tfp.distributions


class Weights(NamedTuple):
    nugget: float
    between: float
    within: float
    interaction: float

    def to_vec(self) -> jnp.DeviceArray:
        return jnp.array([self.nugget, self.between, self.within, self.interaction])

    @staticmethod
    def from_vec(x: jnp.DeviceArray) -> "Weights":
        return Weights(
            nugget=x[0],
            between=x[1],
            within=x[2],
            interaction=x[3],
        )

    def calculate_z(self):
        x = self.to_vec()
        xcumsum = jnp.insert(jnp.cumsum(x[:-1]), 0, 0.0)
        lower = 1 - xcumsum
        z = x / lower
        return z[:-1], lower[:-1]

    def to_stan_simplex(self) -> jnp.DeviceArray:
        z, _ = self.calculate_z()
        K = len(z) + 1
        logit_z = jnp.log(z) - jnp.log(1 - z)
        correction = jnp.log(1 / (K - jnp.arange(K - 1) - 1))
        y = logit_z - correction
        return y

    @staticmethod
    def from_stan_simplex(y) -> "Weights":
        K = len(y) + 1
        correction = jnp.log(1 / (K - jnp.arange(K - 1) - 1))
        z = 1 / (1 + jnp.exp(-(y + correction)))
        x = jnp.empty(K)
        xsum = 0
        for i in range(K - 1):  # not very jaxy but K is small
            if i == 0:
                x = x.at[i].set(z[i])
            else:
                x = x.at[i].set((1 - xsum) * z[i])
            xsum += x[i]
        x = x.at[K - 1].set(1 - jnp.sum(x[:-1]))
        return Weights.from_vec(x)


@dataclass
class M4Data:
    response: jnp.DeviceArray
    design_mat: jnp.DeviceArray
    dist_areas: jnp.DeviceArray
    dist_obs: jnp.DeviceArray

    def num_obs_area(self) -> int:
        return self.dist_obs.shape[0]

    def num_areas(self) -> int:
        return self.dist_areas.shape[0]

    def Z(self) -> jnp.DeviceArray:
        one_m = jnp.ones(self.dist_obs.shape[0])
        Z = jnp.kron(jnp.eye(self.dist_areas.shape[0]), one_m).T
        return Z


@register_dataclass_as_pytree
@dataclass
class M4Params:
    beta: jnp.DeviceArray
    log_variance: jnp.DeviceArray
    log_kappa_b: jnp.DeviceArray
    log_kappa_w: jnp.DeviceArray
    weights_unconst: jnp.DeviceArray
    gamma_b: jnp.DeviceArray

    def get_weights(self) -> Weights:
        return Weights.from_stan_simplex(self.weights_unconst)


class TransformedValues(NamedTuple):
    range_b: float
    range_w: float
    variance: float
    weights: Weights
    error_code: int = 0


class TransformedValsCalculator:
    error_book: ClassVar[dict[int, str]] = {0: "no error"}
    identifier: str = "transformed"

    def __init__(self) -> None:
        self._model: gs.ModelInterface | None = None

    def set_model(self, model: gs.ModelInterface):
        self._model = model

    def has_model(self) -> bool:
        return self._model is not None

    def generate(
        self, prng_key: Any, model_state: Any, epoch: Any
    ) -> TransformedValues:
        if self._model is None:
            raise RuntimeError("Model not set")
        params = self._model.extract_position(
            ["log_kappa_b", "log_kappa_w", "log_variance", "weights_unconst"],
            model_state,
        )

        return {
            "range_b": 3.0 / jnp.exp(params["log_kappa_b"]),
            "range_w": 3.0 / jnp.exp(params["log_kappa_w"]),
            "variance": jnp.exp(params["log_variance"]),
            "weights": Weights.from_stan_simplex(params["weights_unconst"]).to_vec(),
        }


@register_dataclass_as_pytree
@dataclass
class SimplexPriorStan:
    """
    Using a Dirichlet-Prior with Stan's Unit-Simplex transormation.

    https://mc-stan.org/docs/2_22/reference-manual/simplex-transform-section.html
    """

    alphas: Weights

    def log_prob(self, weights: Weights):
        dist = tfd.Dirichlet(concentration=self.alphas.to_vec())
        lpdf = dist.log_prob(weights.to_vec())
        min_weight = weights.to_vec().min()
        lpdf += jax.lax.cond(
            min_weight < 0.001,
            lambda: -jnp.inf,
            lambda: 0.0,
        )

        # Stan Simplex Correction
        z, lower = weights.calculate_z()
        acc = 0.0
        for k in range(len(z)):  # bad jax style but dim(k) is small.
            acc += jnp.log(z[k]) + jnp.log(1 - z[k]) + jnp.log(lower[k])
        return lpdf + acc


@dataclass
class M4Priors:
    beta: jnp.DeviceArray
    log_variance: VarPriorInvGamma | VarPriorHalfNormal | VarPriorHalfCauchy
    weights: SimplexPriorStan

    # log_kappa_b: the interval is chosen by computing practical ranges using the
    # minimum and maximum distances in the data and use them as interval’s extremes
    log_kappa_b: KappaPriorNormal | KappaPriorUniform | KappaPriorNormal2
    log_kappa_w: KappaPriorNormal | KappaPriorUniform | KappaPriorNormal2


@dataclass
class M4:
    engine: gs.Engine
    data: M4Data
    priors: M4Priors


class M4PredData(NamedTuple):
    design_matrix_new: jnp.DeviceArray
    distance_obs_complete: jnp.DeviceArray
    num_new_obs: int
    response: jnp.DeviceArray | None

    def Z(self, num_areas) -> jnp.DeviceArray:
        one_m = jnp.ones(self.num_new_obs)
        Z = jnp.kron(jnp.eye(num_areas), one_m).T
        return Z


class M4PredDataPlot(NamedTuple):
    design_matrix_new: jnp.DeviceArray
    distance_areas_complete: jnp.DeviceArray
    num_new_areas: int
    response: jnp.DeviceArray | None

    def Z(self, num_obs) -> jnp.DeviceArray:
        one_m = jnp.ones(num_obs)
        Z = jnp.kron(jnp.eye(self.num_new_areas), one_m).T
        return Z


def loglik_trick(params: M4Params, data: M4Data):
    """
    using a computational more efficent trick
    by Stegle et al. (2011).
    """
    variance = jnp.exp(params.log_variance)
    weights = params.get_weights()
    gamma_b_blowup = data.Z() @ params.gamma_b
    eta = (
        data.design_mat @ params.beta
        + jnp.sqrt(variance * weights.between) * gamma_b_blowup
    )
    eps = data.response - eta

    corr_b = exponential_covariance(
        h=data.dist_areas, kappa=jnp.exp(params.log_kappa_b), sigma2=1.0
    )
    corr_w = exponential_covariance(
        h=data.dist_obs, kappa=jnp.exp(params.log_kappa_w), sigma2=1.0
    )

    # cov_b = a_1 I + a_2 Sigma^b
    cov_b = weights.interaction * corr_b
    cov_b = cov_b.at[jnp.diag_indices_from(cov_b)].add(weights.within)

    lpdf = stegle_log_density(
        eps, 0.0, weights.nugget, cov_b, corr_w, scale=jnp.sqrt(variance)
    )

    return lpdf


def loglik(params: M4Params, data: M4Data):
    return loglik_trick(params, data)


def log_prior_beta(params: M4Params, priors: M4Priors, data: M4Data):
    beta_nd = tfd.Normal(loc=jnp.repeat(0.0, len(priors.beta)), scale=priors.beta)
    return jnp.sum(beta_nd.log_prob(params.beta))


def log_prior_gamma_b(params: M4Params, priors: M4Priors, data: M4Data):
    sigma_b = exponential_covariance(data.dist_areas, jnp.exp(params.log_kappa_b), 1.0)

    dist = tfd.MultivariateNormalFullCovariance(loc=0.0, covariance_matrix=sigma_b)
    return dist.log_prob(params.gamma_b)


def log_post(params: M4Params, priors: M4Priors, data: M4Data):
    lpdf = 0.0
    lpdf += loglik(params, data)
    lpdf += log_prior_beta(params, priors, data)
    lpdf += log_prior_gamma_b(params, priors, data)

    lpdf += priors.log_variance.log_prob(params.log_variance)
    lpdf += priors.weights.log_prob(Weights.from_stan_simplex(params.weights_unconst))

    lpdf += priors.log_kappa_b.log_prob(params.log_kappa_b)
    lpdf += priors.log_kappa_w.log_prob(params.log_kappa_w)

    return lpdf


def calculator_loglik(params: M4Params, data: M4Data) -> dict[str, float]:
    # log_lik = loglik(params, data)

    Z = data.Z()
    variance = jnp.exp(params.log_variance)
    weights = params.get_weights()

    eta = data.design_mat @ params.beta
    eps = data.response - eta

    corr_b = exponential_covariance(
        h=data.dist_areas, kappa=jnp.exp(params.log_kappa_b), sigma2=1.0
    )
    corr_w = exponential_covariance(
        h=data.dist_obs, kappa=jnp.exp(params.log_kappa_w), sigma2=1.0
    )

    # corr_b_plus = a_1 I + a_2 Sigma^b
    corr_b_plus = weights.interaction * corr_b
    corr_b_plus = corr_b_plus.at[jnp.diag_indices_from(corr_b_plus)].add(weights.within)

    cov = weights.between * (Z @ corr_b @ Z.T)
    cov = cov + jnp.kron(corr_b_plus, corr_w)
    cov = cov.at[jnp.diag_indices_from(cov)].add(weights.nugget)
    cov = variance * cov

    dist = tfd.MultivariateNormalFullCovariance(
        loc=jnp.zeros_like(eps), covariance_matrix=cov
    )

    log_lik = dist.log_prob(eps)

    return {"log_lik": log_lik, "lik": jnp.exp(log_lik)}


def ll_marginal(params: M4Params, data: M4Data) -> dict[str, float]:
    raise NotImplementedError("ll_marginal is not implemented")
    # return {"ll_marginal": lpdf}


def draw_gamma_b(rng_key, params: M4Params, priors: M4Priors, data: M4Data):
    """
    somewhere in the good notes document
    """

    # calculate Q. that is the covariance matrix of the FC.
    weights = params.get_weights()
    variance = jnp.exp(params.log_variance)
    sigma2 = weights.nugget / weights.between

    corr_b = exponential_covariance(
        h=data.dist_areas, kappa=jnp.exp(params.log_kappa_b), sigma2=1.0
    )
    corr_w = exponential_covariance(
        h=data.dist_obs, kappa=jnp.exp(params.log_kappa_w), sigma2=1.0
    )

    mat1_v0 = (weights.interaction / weights.between) * corr_b
    mat1 = mat1_v0.at[jnp.diag_indices_from(mat1_v0)].add(
        weights.within / weights.between
    )
    mat2 = corr_w
    mat1_evd = jnp.linalg.eigh(mat1)
    mat2_evd = jnp.linalg.eigh(mat2)

    Z = data.Z()
    Q = jnp.linalg.inv(corr_b) + stegle_mat(Z, Z, sigma2, mat1_evd, mat2_evd)

    # calculate Q^{-1} u, that is the mean of the FC
    y_wiggle = (data.response - data.design_mat @ params.beta) / jnp.sqrt(
        variance * weights.between
    )
    u = stegle_mat(Z, y_wiggle, sigma2, mat1_evd, mat2_evd)
    Q_inv = jnp.linalg.inv(Q)
    loc_fc = Q_inv @ u

    # draw new gamma
    dist = tfd.MultivariateNormalFullCovariance(loc=loc_fc, covariance_matrix=Q_inv)
    gamma_b_new = dist.sample(seed=rng_key)

    return {"gamma_b": gamma_b_new}


def pred_obs_trick(
    rng_key, params: M4Params, data: M4Data, pred_data: M4PredData
) -> dict[str, jnp.DeviceArray]:
    """predict within plot observations"""

    # for the derivation see notes in GoodNotes

    num_new_obs = pred_data.num_new_obs
    variance = jnp.exp(params.log_variance)
    kappa_b = jnp.exp(params.log_kappa_b)
    kappa_w = jnp.exp(params.log_kappa_w)
    weights = Weights.from_stan_simplex(params.weights_unconst)
    corr_b = exponential_covariance(h=data.dist_areas, kappa=kappa_b, sigma2=1.0)
    corr_w_full = exponential_covariance(
        h=pred_data.distance_obs_complete, kappa=kappa_w, sigma2=1.0
    )
    corr_w_00 = corr_w_full[:num_new_obs, :num_new_obs]
    corr_w_01 = corr_w_full[:num_new_obs, num_new_obs:]
    corr_w_11 = corr_w_full[num_new_obs:, num_new_obs:]

    assert corr_w_00.shape[0] == num_new_obs
    assert corr_w_00.shape[1] == num_new_obs
    assert corr_w_11.shape[1] == data.num_obs_area()
    assert corr_w_11.shape[0] == data.num_obs_area()
    assert corr_w_01.shape[0] == num_new_obs
    assert corr_w_01.shape[1] == data.num_obs_area()

    # 1 refers to old, 0 to new observations
    mu_1 = (
        data.design_mat @ params.beta
        + jnp.sqrt(variance * weights.between) * data.Z() @ params.gamma_b
    )
    mu_0 = (
        pred_data.design_matrix_new @ params.beta
        + jnp.sqrt(variance * weights.between)
        * pred_data.Z(data.num_areas())
        @ params.gamma_b
    )

    # reorder
    num_plots = data.num_areas()
    mu_s_1 = mu_1.reshape(num_plots, -1).T.reshape(-1)
    mu_s_0 = mu_0.reshape(num_plots, -1).T.reshape(-1)
    y_s_1 = data.response.reshape(num_plots, -1).T.reshape(-1)

    # calc block entries of covariance matrix
    rhs_kron = (
        weights.between * jnp.eye(data.num_areas()) + weights.interaction * corr_b
    )
    sigma_00 = jnp.kron(rhs_kron, corr_w_00)
    sigma_00 = sigma_00.at[jnp.diag_indices_from(sigma_00)].add(weights.nugget)

    sigma_01 = jnp.kron(rhs_kron, corr_w_01)

    # don't forget to add the variance
    sigma_00 = variance * sigma_00
    sigma_01 = variance * sigma_01

    # now use stegle to compute the products with sigma_11_inv
    mu_s_0_cond = mu_s_0 + stegle_mat(
        sigma_01.T,
        y_s_1 - mu_s_1,
        variance * weights.nugget,
        variance * rhs_kron,
        corr_w_11,
    )
    cov_s_0_cond = sigma_00 - stegle_mat(
        sigma_01.T,
        sigma_01.T,
        variance * weights.nugget,
        variance * rhs_kron,
        corr_w_11,
    )

    dist = tfd.MultivariateNormalFullCovariance(
        loc=mu_s_0_cond, covariance_matrix=cov_s_0_cond
    )
    sample_s_0_cond = dist.sample(seed=rng_key)

    # reorder
    mu_0_cond = mu_s_0_cond.reshape(-1, num_plots).T.reshape(-1)
    sample_0_cond = sample_s_0_cond.reshape(-1, num_plots).T.reshape(-1)

    if pred_data.response is not None:
        mse_sample_0_cond = jnp.mean((pred_data.response - sample_0_cond) ** 2)
        mse_mu_0_cond = jnp.mean((pred_data.response - mu_0_cond) ** 2)
    else:
        mse_sample_0_cond = None
        mse_mu_0_cond = None

    return {
        "sample_0_cond": sample_0_cond,
        "mu_0_cond": mu_0_cond,
        "mse_sample_0_cond": mse_sample_0_cond,
        "mse_mu_0_cond": mse_mu_0_cond,
    }


def pred_plot(
    rng_key, params: M4Params, data: M4Data, pred_data_plot: M4PredDataPlot
) -> dict[str, jnp.DeviceArray]:
    """predict within plot observations"""
    Z_pred = pred_data_plot.Z(data.num_obs_area())
    Z = data.Z()

    # for the derivation see notes in GoodNotes
    num_new_areas = pred_data_plot.num_new_areas
    variance = jnp.exp(params.log_variance)
    kappa_b = jnp.exp(params.log_kappa_b)
    kappa_w = jnp.exp(params.log_kappa_w)
    weights = Weights.from_stan_simplex(params.weights_unconst)
    corr_b_full = exponential_covariance(
        h=pred_data_plot.distance_areas_complete, kappa=kappa_b, sigma2=1.0
    )
    corr_w = exponential_covariance(h=data.dist_obs, kappa=kappa_w, sigma2=1.0)
    corr_b_00 = corr_b_full[:num_new_areas, :num_new_areas]
    corr_b_01 = corr_b_full[:num_new_areas, num_new_areas:]
    corr_b_11 = corr_b_full[num_new_areas:, num_new_areas:]

    assert corr_b_00.shape[0] == num_new_areas
    assert corr_b_00.shape[1] == num_new_areas
    assert corr_b_11.shape[1] == data.num_areas()
    assert corr_b_11.shape[0] == data.num_areas()
    assert corr_b_01.shape[0] == num_new_areas
    assert corr_b_01.shape[1] == data.num_areas()

    # 1 refers to old, 0 to new observations
    mu_1 = data.design_mat @ params.beta
    mu_0 = pred_data_plot.design_matrix_new @ params.beta

    # no need to reorder
    mu_s_1 = mu_1
    mu_s_0 = mu_0
    y_s_1 = data.response

    # calc block entries of covariance matrix
    corr_w = weights.within * corr_w
    sigma_part2 = corr_w.at[jnp.diag_indices_from(corr_w)].add(weights.nugget)
    sigma_00 = weights.between * (
        Z_pred @ corr_b_00 @ Z_pred.T
    ) + weights.interaction * (jnp.kron(corr_b_00, corr_w))
    sigma_11 = weights.between * (Z @ corr_b_11 @ Z.T) + weights.interaction * (
        jnp.kron(corr_b_11, corr_w)
    )
    sigma_01 = weights.between * (Z_pred @ corr_b_01 @ Z.T) + weights.interaction * (
        jnp.kron(corr_b_01, corr_w)
    )

    sigma_00 = sigma_00 + jnp.kron(jnp.eye(num_new_areas), sigma_part2)
    sigma_11 = sigma_11 + jnp.kron(jnp.eye(data.num_areas()), sigma_part2)

    sigma_00 = variance * sigma_00
    sigma_11 = variance * sigma_11
    sigma_01 = variance * sigma_01

    sigma_11_inv = jnp.linalg.inv(sigma_11)

    # now use stegle to compute the products with sigma_11_inv
    mu_s_0_cond = mu_s_0 + sigma_01 @ sigma_11_inv @ (y_s_1 - mu_s_1)
    cov_s_0_cond = sigma_00 - sigma_01 @ sigma_11_inv @ sigma_01.T

    dist = tfd.MultivariateNormalFullCovariance(
        loc=mu_s_0_cond, covariance_matrix=cov_s_0_cond
    )
    sample_s_0_cond = dist.sample(seed=rng_key)

    # no need to reorder
    mu_0_cond = mu_s_0_cond
    sample_0_cond = sample_s_0_cond

    if pred_data_plot.response is not None:
        mse_sample_0_cond = jnp.mean((pred_data_plot.response - sample_0_cond) ** 2)
        mse_mu_0_cond = jnp.mean((pred_data_plot.response - mu_0_cond) ** 2)
    else:
        mse_sample_0_cond = None
        mse_mu_0_cond = None

    return {
        "sample_0_cond": sample_0_cond,
        "mu_0_cond": mu_0_cond,
        "mse_sample_0_cond": mse_sample_0_cond,
        "mse_mu_0_cond": mse_mu_0_cond,
    }


class M4Builder:
    """
    builds a M4 object
    """

    def __init__(self, response, design_mat, loc_areas, loc_obs, log_score=False):
        assert loc_areas.ndim == 2
        assert loc_obs.ndim == 2
        assert response.ndim == 1
        assert design_mat.ndim == 2
        assert design_mat.shape[0] == response.shape[0]

        if design_mat.shape[1] != jnp.linalg.matrix_rank(design_mat):
            raise RuntimeError("The design matrix is not of full column rank.")

        # default values
        self.seed = 1
        self.num_chains = 4
        self.dur_posterior = 1000
        self.dur_warmup = 1000
        self.dur_term = 200
        self.kernels = []
        self.epochs = None
        self.calculate_ll_marginal = False
        self.keep_gamma_b_const = False
        self.track_gamma_b = False
        self.pred_data: None | M4PredData = None
        self.pred_data_plot: None | M4PredDataPlot = None

        self.log_score = log_score
        self.response = response
        self.design_mat = design_mat
        self.loc_areas = loc_areas
        self.loc_obs = loc_obs

        dist_areas = spat.distance_matrix(loc_areas, loc_areas)
        dist_obs = spat.distance_matrix(loc_obs, loc_obs)

        self.data = M4Data(
            jnp.asarray(response),
            jnp.asarray(design_mat),
            jnp.asarray(dist_areas),
            jnp.asarray(dist_obs),
        )
        self.init_params = M4Params(
            beta=jnp.zeros(design_mat.shape[1]),
            log_variance=jnp.array(jnp.log(1.0)),
            log_kappa_b=jnp.array(jnp.log(4.0)),
            log_kappa_w=jnp.array(jnp.log(4.0)),
            weights_unconst=Weights(0.25, 0.25, 0.25, 0.25).to_stan_simplex(),
            gamma_b=jnp.zeros(dist_areas.shape[1]),
        )

        self.priors = M4Priors(
            beta=jnp.append(100.0, jnp.repeat(10.0, design_mat.shape[1] - 1)),
            log_variance=VarPriorInvGamma(0.001, 0.001),
            log_kappa_b=KappaPriorUniform(1.0, 300.0),
            log_kappa_w=KappaPriorUniform(1.0, 300.0),
            weights=SimplexPriorStan(Weights(1.0, 1.0, 1.0, 1.0)),
        )

        self.set_default_kernel()

    def set_pred_data(
        self, design_matrix_new, loc_obs_old, loc_obs_new, response_new=None
    ):
        # create distance_matrix
        D = spat.distance_matrix(loc_obs_old, loc_obs_old)
        A = spat.distance_matrix(loc_obs_new, loc_obs_new)
        C = spat.distance_matrix(loc_obs_old, loc_obs_new)
        distance_matrix = jnp.block([[A, C.T], [C, D]])

        self.pred_data = M4PredData(
            design_matrix_new, distance_matrix, loc_obs_new.shape[0], response_new
        )

    def set_pred_data_plot(
        self, design_matrix_new, loc_areas_old, loc_areas_new, response_new=None
    ):
        # create distance_matrix
        D = spat.distance_matrix(loc_areas_old, loc_areas_old)
        A = spat.distance_matrix(loc_areas_new, loc_areas_new)
        C = spat.distance_matrix(loc_areas_old, loc_areas_new)
        distance_matrix = jnp.block([[A, C.T], [C, D]])

        self.pred_data_plot = M4PredDataPlot(
            design_matrix_new, distance_matrix, loc_areas_new.shape[0], response_new
        )

    def set_default_kernel(self):
        self.kernels = []

        ker = gs.HMCKernel(
            [
                "beta",
                "weights_unconst",
                "log_variance",
                "log_kappa_b",
                "log_kappa_w",
            ],
        )

        self.kernels.append(ker)

    def build(self) -> M4:
        logp = partial(log_post, data=self.data, priors=self.priors)
        # minfo = partial(ll_marginal, data=self.data)

        builder = gs.EngineBuilder(self.seed, self.num_chains)
        for ker in self.kernels:
            builder.add_kernel(ker)

        if not self.keep_gamma_b_const:
            draw = partial(draw_gamma_b, priors=self.priors, data=self.data)
            gk = gs.GibbsKernel(["gamma_b"], draw)
            builder.add_kernel(gk)

        builder.set_model(DataClassConnector(logp))
        builder.set_initial_values(self.init_params, False)
        if self.epochs is not None:
            builder.set_epochs(self.epochs)
        else:
            builder.set_duration(self.dur_warmup, self.dur_posterior, self.dur_term)
        builder.store_kernel_states = False

        if not self.track_gamma_b:
            builder.positions_excluded = ["gamma_b"]

        builder.add_quantity_generator(TransformedValsCalculator())

        if self.log_score:
            qg_ll = quantgen(
                lambda key, params, mi, epoch: calculator_loglik(params, self.data),
                "log_lik",
            )
            builder.add_quantity_generator(qg_ll)

        # if self.calculate_ll_marginal:
        #     qg0 = quantgen(
        #         lambda key, params, mi, epoch: ll_marginal(params, self.data),
        #         "ll_marginal",
        #     )
        #     builder.add_quantity_generator(qg0)

        if self.pred_data is not None:
            qg1 = quantgen(
                lambda key, params, mi, epoch: pred_obs_trick(
                    key, params, self.data, self.pred_data
                ),
                "predictions",
            )
            builder.add_quantity_generator(qg1)

        if self.pred_data_plot is not None:
            qg1 = quantgen(
                lambda key, params, mi, epoch: pred_plot(
                    key, params, self.data, self.pred_data_plot
                ),
                "predictions",
            )
            builder.add_quantity_generator(qg1)

        engine = builder.build()

        return M4(engine, self.data, self.priors)


def create_index_subsets(rng_key, num_areas, num_obs_area, npoints_leave_out):
    """
    creates the index sets to select a subset such that obs at the same
    locations within a plot are selected or deselected.
    to be used with predictions in model4.
    returns a lost of tuples.
    - tuple[0]: old (response)
    - tuple[1]: new (response)
    - tuple[2]: old (loc_obs)
    - tuple[3]: new (loc_obs)
    """

    npoints_lvl2_old = num_obs_area - npoints_leave_out

    rs = jax.random.shuffle(
        rng_key,
        jnp.arange(num_obs_area),
    )

    if len(rs) % npoints_leave_out > 0:
        leave_out_idxs = rs[: -(len(rs) % npoints_leave_out)].reshape(
            -1, npoints_leave_out
        )
    else:
        leave_out_idxs = rs.reshape(-1, npoints_leave_out)

    idx_obs = jnp.arange(num_obs_area)

    ret_list = []
    for i in range(leave_out_idxs.shape[0]):
        idx_obs_new = leave_out_idxs[i].sort()
        idx_obs_old = jnp.setdiff1d(idx_obs, idx_obs_new)

        idx_old = jnp.repeat(
            jnp.arange(num_areas) * num_obs_area, npoints_lvl2_old
        ) + jnp.tile(idx_obs_old, num_areas)
        idx_new = jnp.repeat(
            jnp.arange(num_areas) * num_obs_area, npoints_leave_out
        ) + jnp.tile(idx_obs_new, num_areas)

        assert len(jnp.intersect1d(idx_old, idx_new)) == 0
        assert len(jnp.unique(jnp.union1d(idx_old, idx_new))) == len(idx_new) + len(
            idx_old
        )
        ret_list.append((idx_old, idx_new, idx_obs_old, idx_obs_new))

    return ret_list


def create_index_subsets_plot(rng_key, num_areas, num_obs_area, npoints_leave_out):
    """
    setting indexes for predictions on new plots.

    to be used with predictions in model4 pred_data_plot.

    """

    npoints_lvl1_old = num_areas - npoints_leave_out

    rs = jax.random.shuffle(
        rng_key,
        jnp.arange(num_areas),
    )

    if len(rs) % npoints_leave_out > 0:
        leave_out_idxs = rs[: -(len(rs) % npoints_leave_out)].reshape(
            -1, npoints_leave_out
        )
    else:
        leave_out_idxs = rs.reshape(-1, npoints_leave_out)

    idx_areas = jnp.arange(num_areas)

    ret_list = []
    for i in range(leave_out_idxs.shape[0]):
        idx_areas_new = leave_out_idxs[i].sort()
        idx_areas_old = jnp.setdiff1d(idx_areas, idx_areas_new)

        idx_old = jnp.repeat(
            jnp.array(idx_areas_old) * num_obs_area, num_obs_area
        ) + jnp.tile(jnp.arange(num_obs_area), npoints_lvl1_old)

        idx_new = jnp.repeat(
            jnp.array(idx_areas_new) * num_obs_area, num_obs_area
        ) + jnp.tile(jnp.arange(num_obs_area), npoints_leave_out)

        assert len(jnp.intersect1d(idx_old, idx_new)) == 0
        assert len(jnp.unique(jnp.union1d(idx_old, idx_new))) == len(idx_new) + len(
            idx_old
        )

        ret_list.append((idx_old, idx_new, idx_areas_old, idx_areas_new))

    return ret_list


def is_posterior(epoch_config) -> bool:
    """tests if a epoch_config is from the posterior phase"""
    return epoch_config.type.POSTERIOR == epoch_config.type
