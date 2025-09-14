import jax
import jax.numpy as jnp
from flax import nnx
import structlog

log = structlog.get_logger()


# x starts with (batch size, features)
# converts to (batch_size, hidden_layer_width)
# each layer does (batch_size, hidden_layer_width) -> (batch_size, hidden_layer_width)
# output layer (batch_size, hidden_layer_width) -> (batch_size, 1)

# first weight (features, hidden_layer_width)
# each inner layer weight is (hidden_layer_width, hidden_layer_width)
# outer layer weight (hidden_layer_width, 1)


class HiddenLayer(nnx.Module):
    """each hidden layer of the model"""

    def __init__(self, hidden_dim: int, key, hidden_activation):
        self.hidden_dim = hidden_dim
        self.w = nnx.Param(
            jax.random.normal(key, (self.hidden_dim, self.hidden_dim))
            * (1 / jnp.sqrt(self.hidden_dim))
        )
        self.b = nnx.Param(jnp.zeros((1, self.hidden_dim)))
        self.hidden_activation = hidden_activation

    def __call__(self, x: jax.Array):
        # x is (batch_size, hidden_layer_width)
        # do the thing (Wx+b)
        log.debug("shape of x", x=x.shape)
        inner = x @ self.w + self.b
        # the nonlinear of hidden_activation
        return self.hidden_activation(inner)  # (batch_size, hidden_layer_width)


class NNXSpiralModel(nnx.Module):
    """model for classification"""

    def __init__(
        self,
        *,
        rngs: nnx.Rngs,
        num_inputs: int,
        num_outputs: int,
        num_layers: int,
        hidden_layer_width: int,
        hidden_activation=nnx.identity,  # choice of nonlinearity in each layer
        output_activation=nnx.identity,
    ):  # choice for output
        key = rngs.params()
        self.num_features = num_inputs
        self.num_layers = num_layers
        self.hidden_layer_width = hidden_layer_width
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation

        self.first_w = nnx.Param(
            jax.random.normal(key, (self.num_features, self.hidden_layer_width))
            * (1 / jnp.sqrt(self.hidden_layer_width))
        )
        self.first_b = nnx.Param(jnp.zeros((1, self.hidden_layer_width)))

        @nnx.vmap(in_axes=0, out_axes=0)
        def MakeHiddenLayer(key: jax.Array):
            return HiddenLayer(
                hidden_dim=self.hidden_layer_width,
                key=key,
                hidden_activation=self.hidden_activation,
            )

        self.keys = jax.random.split(key, self.num_layers - 1)

        self.hidden_layers = MakeHiddenLayer(self.keys)

        self.out_w = nnx.Param(
            jax.random.normal(key, (self.hidden_layer_width, 1))
            * (1 / jnp.sqrt(self.hidden_layer_width))
        )
        self.out_b = nnx.Param(jnp.zeros((1, 1)))

    def __call__(self, x: jax.Array):
        # assume x is (batch_size, num_features)
        # do all the hidden layers
        @nnx.scan(in_axes=(0, nnx.Carry), out_axes=nnx.Carry)
        def forward(layer: HiddenLayer, x):
            x = layer(x)
            return x

        # x is (batch_size, features)
        x_first = x @ self.first_w + self.first_b
        # jax.debug.print("x_first = {}",x_first)
        # jax.debug.print("x_first after act = {}", self.hidden_activation(x_first))
        x_first = self.hidden_activation(x_first)
        # x_first is (batch_size, hidden_layer_width)
        t_inner = forward(self.hidden_layers, x_first)
        # t_inner is (batch_size, hidden_layer_width) after all layers

        # do outer layer
        t_outer = t_inner @ self.out_w + self.out_b  # (batch_size, 1)
        # apply outer output activation function
        # jax.debug.print("t_outer = {}", t_outer)

        # jax.debug.print("t_outer post act = {}", self.output_activation(t_outer))
        return self.output_activation(t_outer)
