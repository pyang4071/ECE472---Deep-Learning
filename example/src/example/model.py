from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx


@dataclass
class LinearModel:
    """Represents a simple linear model."""

    weights: np.ndarray
    bias: float


class NNXLinearModel(nnx.Module):
    """A Flax NNX module for a linear regression model."""

    def __init__(self, *, rngs: nnx.Rngs, num_features: int):
        self.num_features = num_features
        key = rngs.params()
        self.w = nnx.Param(jax.random.normal(key, (self.num_features, 1)))
        self.b = nnx.Param(jnp.zeros((1, 1)))

    def __call__(self, x: jax.Array) -> jax.Array:
        """Predicts the output for a given input."""
        return jnp.squeeze(x @ self.w.value + self.b.value)

    @property
    def model(self) -> LinearModel:
        """Returns the underlying simple linear model."""
        return LinearModel(
            weights=np.array(self.w.value).reshape([self.num_features]),
            bias=np.array(self.b.value).squeeze(),
        )
