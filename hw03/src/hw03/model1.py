import jax
import jax.numpy as jnp
from flax import nnx
import structlog

log = structlog.get_logger()


class Conv2d(nnx.Module):
    """each convolution layer"""

    def __init__(
        self,
        keys,
        l2reg: float,
        dropout_rate: float,
        in_features,
        out_features,
        kernel_size,
        strides,
        padding: str = "SAME",
    ):
        self.rngs = nnx.Rngs(params=keys, dropout=keys)
        self.l2reg = l2reg
        self.padding = padding

        self.layer = nnx.Conv(
            in_features=in_features,
            out_features=out_features,
            kernel_size=kernel_size,
            strides=strides,
            padding=self.padding,
            rngs=self.rngs,
        )
        self.dropout = nnx.Dropout(rate=dropout_rate, rngs=self.rngs)

    def __call__(self, x: jnp.ndarray, train: bool):
        x = self.layer(x)
        x = self.dropout(x, deterministic=not train)
        return x  # (num_samples, ??, ??, out_features)

    def l2loss(self):
        return self.l2reg * jnp.sum(jnp.square(self.layer.kernel.value))


class Classifier2(nnx.Module):
    """model for MNIST classification"""

    def __init__(
        self,
        *,
        input_depth: int,
        layer_depths: list[int],
        layer_kernel_sizes: list[tuple[int, int]],
        strides: list[int],
        num_classes: int,
        dropout_rate: float,
        l2reg: float,
        rngs: nnx.Rngs,
        shape: list[int],
    ):
        keys = rngs.params()
        self.input_depth = input_depth
        self.layer_depths = layer_depths  # output depth of each layer
        self.layer_kernel_sizes = layer_kernel_sizes
        self.num_classes = num_classes
        self.strides = strides

        self.dropout_rate = dropout_rate
        self.l2reg = l2reg

        self.in_features = [self.input_depth] + self.layer_depths[
            :-1
        ]  # all the input depths

        self.keys = jax.random.split(keys, len(self.layer_depths))

        log.debug(
            "conv layers dims",
            depths=self.layer_depths,
            in_features=self.in_features,
            kernels=self.layer_kernel_sizes,
            strides=self.strides,
        )

        self.layers = []

        for i in range(len(self.layer_depths)):
            self.layers.append(
                Conv2d(
                    in_features=self.in_features[i],
                    out_features=self.layer_depths[i],
                    kernel_size=self.layer_kernel_sizes[i],
                    strides=self.strides[i],
                    keys=self.keys[i],
                    l2reg=self.l2reg,
                    dropout_rate=self.dropout_rate,
                )
            )

        log.debug("layers", layers=self.layers)

        log.debug(
            "val",
            in_features=self.in_features,
            layer_depths=self.layer_depths,
            kern_size=self.layer_kernel_sizes,
            strides=self.strides,
        )

        flatten_size = shape[0] * shape[1] * self.layer_depths[-1]

        self.final_layer = nnx.Linear(
            in_features=flatten_size, out_features=self.num_classes, rngs=rngs
        )

    def __call__(self, x: jax.Array, train: bool):
        # assume x is (batch size, 28, 28, 1)
        val = x
        for layers in self.layers:
            val = layers(val, train)
            val = jax.nn.leaky_relu(val)

        val = val.reshape((val.shape[0], -1))  # (batchsize, h*w*d)
        val = self.final_layer(val)
        return val

    def l2loss(self):
        l2_loss = 0
        for layers in self.layers:
            l2_loss += layers.l2loss()
        return l2_loss
