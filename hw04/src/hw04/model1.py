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

    def __call__(self, x: jnp.ndarray):
        x = self.layer(x)
        return x  # (num_samples, h, w, out_features)

    def l2loss(self):
        return self.l2reg * jnp.sum(jnp.square(self.layer.kernel.value))


class GroupNorm(nnx.Module):
    def __init__(self, num_channels: int, num_groups: int, epsilon: float = 1e-5):
        self.gamma = nnx.Param(jnp.ones((num_channels,)))
        self.beta = nnx.Param(jnp.zeros((num_channels,)))

        self.num_groups = num_groups
        self.epsilon = epsilon

    def __call__(self, x: jax.Array) -> jax.Array:
        # x is shape (num_samples, height, width, depth/channels)
        N, H, W, C = x.shape
        x_group = x.reshape(N, H, W, self.num_groups, C // self.num_groups)
        # shape = (num_samples, height, width, num_groups, channels_per_group)

        # normalize per group

        # find the mean and var per group in each sample
        u = jnp.mean(x_group, axis=(1, 2, 4), keepdims=True)
        var = jnp.var(x_group, axis=(1, 2, 4), keepdims=True)
        # shape = (num_samples, 1, 1, num_groups, 1)
        # number of means = num_samples*num_groups_per_sample

        # normalize
        x_group_norm = (x_group - u) / (jnp.sqrt(var + self.epsilon))
        # shape = (num_samples, h, w, groups, chan per group)

        # reshape back to normal
        x_normed = x_group_norm.reshape(N, H, W, C)

        # scale and shift by gamma and beta
        gamma = self.gamma.reshape(1, 1, 1, C)
        beta = self.beta.reshape(1, 1, 1, C)
        # scale and shift per channel

        x_hat = gamma * x_normed + beta

        return x_hat


class ResidualBlock(nnx.Module):
    def __init__(
        self,
        keys,
        l2reg,
        in_features,
        out_features,
        kernel_size,
        strides,
        num_groups,
        activation=jax.nn.leaky_relu,
    ):
        if in_features != out_features or strides != 1:
            # create a 1 by 1 kernel conv layer to make the shortcutted x into the right shape
            # so we can do the thing, x+fx
            x_key, fx_key = jax.random.split(keys, 2)
            self.conv_layer_x = Conv2d(
                in_features=in_features,
                out_features=out_features,
                kernel_size=(1, 1),
                strides=strides,
                keys=x_key,
                l2reg=l2reg,
            )
            self.con_x = True
        else:
            fx_key = keys
            self.con_x = False

        fx_key_1, fx_key_2 = jax.random.split(fx_key)

        self.group_norm_1 = GroupNorm(in_features, num_groups)

        self.conv_layer_1 = Conv2d(
            in_features=in_features,
            out_features=out_features,
            kernel_size=kernel_size,
            strides=strides,
            keys=fx_key_1,
            l2reg=l2reg,
        )

        self.group_norm_2 = GroupNorm(out_features, num_groups)

        self.conv_layer_2 = Conv2d(
            in_features=out_features,
            out_features=out_features,
            kernel_size=kernel_size,
            strides=1,
            keys=fx_key_2,
            l2reg=l2reg,
        )

        self.activation = activation

    def __call__(self, x: jax.Array) -> jax.Array:
        # x is shape(num_samples, height, width, in_features)
        fx = self.group_norm_1(x)
        fx = self.activation(fx)
        fx = self.conv_layer_1(fx)
        fx = self.group_norm_2(fx)
        fx = self.activation(fx)
        fx = self.conv_layer_2(fx)
        # fx is shape(num_samples, height, width, out_features)

        # fix x shape if needed
        if self.con_x:
            x = self.conv_layer_x(x)

        return x + fx

    def l2loss(self):
        l2loss = self.conv_layer_1.l2loss()
        l2loss = l2loss + self.conv_layer_2.l2loss()

        if self.con_x:
            l2loss = l2loss + self.conv_layer_x.l2loss()

        return l2loss


class Classifier(nnx.Module):
    """model: for loop version"""

    def __init__(
        self,
        *,
        rngs: nnx.Rngs,
        input_depth: int,
        input_shape: tuple[int, int],
        layer_depths: list[int],
        layer_kernel_size: list[tuple[int, int]],
        num_groups: int,
        strides: list[int],
        num_classes: int = 10,
        l2reg: float = 0.001,
    ):
        keys = rngs.params()
        self.input_depth = input_depth
        self.layer_depths = layer_depths  # output depth of each layer
        self.layer_kernel_size = layer_kernel_size
        self.num_classes = num_classes
        self.num_groups = num_groups
        self.strides = strides

        self.l2reg = l2reg

        self.in_features = [self.input_depth] + self.layer_depths[:-1]
        self.keys = jax.random.split(keys, len(self.layer_depths))

        self.first_layer = Conv2d(
            keys=self.keys[0],
            l2reg=self.l2reg,
            in_features=self.in_features[0],
            out_features=self.layer_depths[0],
            kernel_size=self.layer_kernel_size[0],
            strides=self.strides[0],
        )
        self.first_group_norm = GroupNorm(
            num_channels=self.layer_depths[0],
            num_groups=self.num_groups[0],
        )
        self.first_activation = jax.nn.leaky_relu

        self.layers = nnx.List([])

        for i in range(1, len(self.layer_depths)):
            self.layers.append(
                ResidualBlock(
                    keys=self.keys[i],
                    l2reg=self.l2reg,
                    in_features=self.in_features[i],
                    out_features=self.layer_depths[i],
                    kernel_size=self.layer_kernel_size[i],
                    strides=self.strides[i],
                    num_groups=self.num_groups[i],
                )
            )

        # perform an average global pooling in __call__
        # x becomes shape (num_samples, 1, 1, self.layer_depths[-1])

        self.final_layer = nnx.Linear(
            in_features=self.layer_depths[-1], out_features=self.num_classes, rngs=rngs
        )

    def __call__(self, x: jax.Array):
        # assume x is (batch size, 28, 28, 3)
        val = x
        val = self.first_layer(val)
        val = self.first_group_norm(val)
        val = self.first_activation(val)
        for layers in self.layers:
            val = layers(val)
        # global pooling
        val = jnp.mean(val, axis=(1, 2))
        val = val.reshape((val.shape[0], -1))
        val = self.final_layer(val)
        return val

    def l2loss(self):
        l2_loss = self.first_layer.l2loss()

        for layers in self.layers:
            l2_loss += layers.l2loss()
        return l2_loss
