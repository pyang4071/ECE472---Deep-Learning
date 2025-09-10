from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx


@dataclass
class ModelParams:
    """model with params"""
    weights: np.ndarray
    means: np.ndarray
    variances: np.ndarray
    bias: float

class NNXModel(nnx.Module):
    def __init__(self, *, rngs: nnx.Rngs, num_basis: int):
        key = rngs.params()
        self.num_basis = num_basis
        self.w = nnx.Param(jax.random.normal(key, (self.num_basis, 1)) * 0.5)
        self.u = nnx.Param(jax.random.uniform(key, (1, self.num_basis), minval=0.0, maxval=1.0))
        self.sigma = nnx.Param(jax.random.normal(key, (1, self.num_basis)) * 0.1)
        self.b = nnx.Param(jnp.zeros((1,1)))
    
    def __call__(self, x: jax.Array):
        """Return the prediction of the given x"""
        # assume x is (batch size,)
        x = x.reshape(-1,1) # change x to (batchsize, 1)
        # w is (num_basis,1)
        # u is (1, num_basis)
        # sigma is (1, num_basis)
        # phi = exp(-(x-u)^2/sigma^2)
        # each row is one measurement
        exp_num = -1*(jnp.square(x-self.u)) #(batchsize, num_basis)
        exp_denom = jnp.square(self.sigma) 
        exp_all = exp_num / exp_denom
        exp = jnp.exp(exp_all) # (batchsize, num_basis)
        y_hat_no_bias = exp @ self.w # (batchsize, 1)
        return jnp.squeeze(y_hat_no_bias + self.b)

    @property 
    def model_params(self) -> ModelParams:
        """return model params"""
        return ModelParams(
            weights=np.array(self.w.value).reshape([self.num_basis]),
            means=np.array(self.u.value).reshape([self.num_basis]),
            variances=np.array(self.sigma.value).reshape([self.num_basis]),
            bias=np.array(self.b.value).squeeze(),
        )

