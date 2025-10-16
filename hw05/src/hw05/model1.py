import jax
import jax.numpy as jnp
from flax import nnx
import structlog


log = structlog.get_logger()


class HiddenLayer(nnx.Module):
    """each hidden layer of the model"""

    def __init__(self, num_inputs: int, hidden_dim: int, key, hidden_activation):
        self.hidden_dim = hidden_dim
        self.w = nnx.Param(
            jax.random.normal(key, (num_inputs, self.hidden_dim))
            * (jnp.sqrt(2 / num_inputs))
        )
        self.b = nnx.Param(jnp.zeros((1, self.hidden_dim)))
        self.hidden_activation = hidden_activation

    def __call__(self, x: jax.Array):
        # x is (batch_size, hidden_layer_width)
        # do the thing (Wx+b)
        log.debug("shape of x", x=x.shape)
        inner = x @ self.w + self.b

        return self.hidden_activation(inner)  # (batch_size, hidden_layer_width)

    def l2loss(self):
        return jnp.sum(jnp.square(self.w)) + jnp.sum(jnp.square(self.b))


class Classifier(nnx.Module):
    """model for classification"""

    def __init__(
        self,
        rngs,
        max_token,
        embed_depth,
        num_outputs,
        num_layers,
        hidden_layer_width,
        hidden_activation=jax.nn.leaky_relu,
        l2reg: float = 0.0001,
    ):
        key, embed_key = jax.random.split(rngs.params(), 2)
        self.keys = jax.random.split(key, num_layers)
        self.l2reg = l2reg

        # embed
        # (num_samples, max_len)
        self.embed_layer = nnx.Param(
            jax.random.normal(embed_key, (max_token, embed_depth))
            * jnp.sqrt(2 / max_token)
        )

        self.first_layer = HiddenLayer(
            key=self.keys[0],
            hidden_dim=hidden_layer_width,
            num_inputs=embed_depth,
            hidden_activation=hidden_activation,
        )

        @nnx.vmap(in_axes=0, out_axes=0)
        def MakeHiddenLayer(key: jax.Array):
            return HiddenLayer(
                hidden_dim=hidden_layer_width,
                key=key,
                num_inputs=hidden_layer_width,
                hidden_activation=hidden_activation,
            )

        self.hidden_layers = MakeHiddenLayer(self.keys[1:])

        self.final_layer = nnx.Linear(
            in_features=hidden_layer_width, out_features=num_outputs, rngs=rngs
        )

    def __call__(self, x):
        @nnx.scan(in_axes=(0, nnx.Carry), out_axes=nnx.Carry)
        def forward(layer: HiddenLayer, x):
            x = layer(x)
            return x

        x = self.embed_layer[x]
        # shape (num_samples, token_len, embed_depth)

        x = jnp.mean(x, axis=1)
        # shape (num_samples, embed_depth)

        x = self.first_layer(x)
        # shape (num_samples, hidden_layer_width)

        x = forward(self.hidden_layers, x)
        # shape (num_samples, hidden_layer_width)

        # last layer - linear
        x = self.final_layer(x)
        # shape (num_samples, num_outputs)

        return x

    def l2loss(self):
        l2loss = self.first_layer.l2loss()

        @nnx.scan(in_axes=(0, nnx.Carry), out_axes=nnx.Carry)
        def loss(layer: HiddenLayer, loss):
            loss = layer.l2loss()
            return loss

        l2loss += loss(self.hidden_layers, l2loss)
        return self.l2reg * l2loss
