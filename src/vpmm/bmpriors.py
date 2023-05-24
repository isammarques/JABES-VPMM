from dataclasses import dataclass

import jax.numpy as jnp
import jax.scipy.stats
import tensorflow_probability.substrates.jax as tfp
from liesel.goose.pytree import register_dataclass_as_pytree

tfd = tfp.distributions


@register_dataclass_as_pytree
@dataclass
class KappaPriorUniform:
    lower: float
    upper: float

    def log_prob(self, log_kappa):
        dist = tfd.Uniform(
            low=self.lower,
            high=self.upper,
        )

        var = jnp.exp(log_kappa)
        lpdf = dist.log_prob(var)
        lpdf += log_kappa  # log space correction
        return lpdf


@register_dataclass_as_pytree
@dataclass
class KappaPriorNormal:
    loc: float
    scale: float

    def log_prob(self, log_kappa):
        dist = tfd.Normal(
            loc=self.loc,
            scale=self.scale,
        )

        # no log space correction
        # since the parameter is log_kappa
        lpdf = dist.log_prob(log_kappa)
        return lpdf


@register_dataclass_as_pytree
@dataclass
class KappaPriorNormal2:
    rho1: float
    rho2: float
    quantile1: float
    quantile2: float

    def get_loc_and_scale(self) -> tuple[float, float]:
        norm = jax.scipy.stats.norm
        sigma = (jnp.log(self.rho1) - jnp.log(self.rho2)) / (
            norm.ppf(self.quantile1) - norm.ppf(self.quantile2)
        )
        mu = jnp.log(3.0) - jnp.log(self.rho1) + sigma * norm.ppf(self.quantile1)

        return mu, sigma

    def log_prob(self, log_kappa):
        loc, scale = self.get_loc_and_scale()
        dist = tfd.Normal(
            loc=loc,
            scale=scale,
        )

        # no log space correction
        # since the parameter is log_kappa
        lpdf = dist.log_prob(log_kappa)
        return lpdf


@register_dataclass_as_pytree
@dataclass
class VarPriorInvGamma:
    concentration: float
    scale: float

    def log_prob(self, log_var):
        dist = tfd.InverseGamma(concentration=self.concentration, scale=self.scale)
        var = jnp.exp(log_var)
        lpdf = dist.log_prob(var)
        lpdf += log_var  # log space correction
        return lpdf


@register_dataclass_as_pytree
@dataclass
class VarPriorHalfNormal:
    scale: float

    def log_prob(self, log_var):
        dist = tfd.HalfNormal(scale=self.scale)

        sd = jnp.exp(0.5 * log_var)
        lpdf = dist.log_prob(sd)
        lpdf += 0.5 * log_var - jnp.log(2.0)  # change of variable correction
        return lpdf


@register_dataclass_as_pytree
@dataclass
class VarPriorHalfCauchy:
    scale: float

    def log_prob(self, log_var):
        dist = tfd.HalfCauchy(loc=0.0, scale=self.scale)

        sd = jnp.exp(0.5 * log_var)
        lpdf = dist.log_prob(sd)
        lpdf += 0.5 * log_var - jnp.log(2.0)  # change of variable correction
        return lpdf
