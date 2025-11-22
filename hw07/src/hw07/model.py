import jax
import jax.numpy as jnp
from flax import nnx
import structlog

log = structlog.get_logger()


class HiddenLayer(nnx.Module):
    """each hidden layer of the classifier model"""

    def __init__(self, init_w, hidden_activation):
        _, out_dim = init_w.shape
        self.w = nnx.Param(init_w)
        self.b = nnx.Param(jnp.zeros((1, out_dim)))
        self.hidden_activation = hidden_activation

    def __call__(self, x: jax.Array):
        # x is (batch_size,in_dim)
        # do the thing (Wx+b)
        inner = x @ self.w + self.b

        return self.hidden_activation(inner)  # (batch, out_dim)

    def l2loss(self):
        return jnp.sum(jnp.square(self.w)) + jnp.sum(jnp.square(self.b))


class NNXSpiralModel(nnx.Module):
    """model for classification"""

    def __init__(
        self,
        *,
        key: jax.Array,
        in_dim: int,
        layer_dim: list[int],
        hidden_activation=jax.nn.relu,
        output_activation=jax.nn.sigmoid,
    ):
        keys = jax.random.split(key, len(layer_dim) + 1)

        in_dims = [in_dim] + layer_dim[:-1]

        layers = []
        for i in range(len(layer_dim)):
            w = jax.random.normal(keys[i], (in_dims[i], layer_dim[i])) * jnp.sqrt(
                2 / in_dims[i]
            )
            layers.append(HiddenLayer(w, hidden_activation))

        self.mlp_layers = nnx.List(layers)

        w = jax.random.normal(keys[-1], (layer_dim[-1], 1)) * jnp.sqrt(
            2 / layer_dim[-1]
        )
        self.fc = HiddenLayer(w, output_activation)

    def get_features(self, x: jax.Array):
        for i in range(len(self.mlp_layers)):
            x = self.mlp_layers[i](x)

        return x

    def __call__(self, x: jax.Array):
        # assume x is (batch_size, num_features)
        features = self.get_features(x)
        prob = self.fc(features)

        return prob

    def l2loss(self):
        l2 = 0.0
        for i in range(len(self.mlp_layers)):
            l2 += self.mlp_layers[i].l2loss()

        l2 += self.fc.l2loss()
        return l2


class SparseEncoder(nnx.Module):
    """sparse encoder to evaluate classifier"""

    def __init__(
        self,
        *,
        key: jax.Array,
        in_dim: int,
        hidden_dim: int,
        norm_val: int,
    ):
        # keys = jax.random.split(key, 2)

        w_dec = jax.random.normal(key, (hidden_dim, in_dim))

        # each column l2 norm = 0.1
        for i in range(in_dim):
            col = w_dec[:, i]
            # normalize
            col = col / jnp.linalg.norm(col)
            # equal norm to chosen value
            col = col * norm_val
            w_dec = w_dec.at[:, i].set(col)

        w_enc = w_dec.T

        self.enc = HiddenLayer(w_enc, jax.nn.relu)
        self.dec = HiddenLayer(w_dec, jax.nn.identity)

    def get_features(self, x):
        x = self.enc(x)
        return x

    def __call__(self, x):
        features = self.get_features(x)
        output = self.dec(features)
        return features, output, self.dec.w
