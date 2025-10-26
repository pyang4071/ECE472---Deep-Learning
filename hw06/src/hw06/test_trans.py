import jax
import jax.numpy as jnp
import optax
import structlog

from flax import nnx

from .config import AppSettings
from .model import Transformer, Multihead_attention

log = structlog.get_logger()


class test_transformer:
    def __init__(self, settings: AppSettings):
        key = jax.random.PRNGKey(settings.random_seed)
        self.data_key, self.model_key = jax.random.split(key)
        # self.np_rng = np.random.default_rng(np.array(self.data_key))

        self.mha = Multihead_attention(
            key=self.model_key,
            num_head=settings.model.num_head,
            embed_depth=settings.model.embed_depth,
        )

        self.transformer = Transformer(
            key=self.model_key,
            num_head=settings.model.num_head,
            embed_depth=settings.model.embed_depth,
            ff_dim=settings.model.ff_dim,
        )

        self.x_test = jax.random.normal(
            self.data_key,
            (
                settings.testing.batch_size,
                settings.testing.seq_length,
                settings.model.embed_depth,
            ),
        )

        self.mha_output_shape()
        self.mha_mask()
        self.mha_grad()
        self.trans_output_shape()
        self.trans_grad()

    def mha_output_shape(self):
        out = self.mha(self.x_test, self.x_test, self.x_test, False)

        assert out.shape == self.x_test.shape
        log.info("transformer output shape correct")

    def mha_mask(self):
        _, atten_w = self.mha(self.x_test, self.x_test, self.x_test, True, True)
        # (batch, num_head, seq_q, seq_k)
        # check if masked regions are close to 0
        mask = jnp.tril(jnp.ones(atten_w[0][0].shape))
        casual_mask = 1 - mask  # 0 is pass

        masked = atten_w * casual_mask
        assert jnp.max(masked) < 1e-4
        log.info("masking works")

    def mha_grad(self):
        _, w_init = self.mha(self.x_test, self.x_test, self.x_test, False, True)

        optimizer = nnx.Optimizer(self.mha, optax.adam(0.1), wrt=nnx.Param)

        def loss_fn(model, x):
            out = model(x, x, x, False, False)
            return jnp.sum(out)

        loss, grads = nnx.value_and_grad(loss_fn)(self.mha, self.x_test)

        # get all the gradients - gradients are a tree
        leaves = jax.tree_util.tree_leaves(grads)
        for leaf in leaves:
            # check that at least each parameter recieves some gradient
            mini_grad = jnp.any(leaf != 0)
            assert mini_grad

        optimizer.update(self.mha, grads)
        _, w_new = self.mha(self.x_test, self.x_test, self.x_test, False, True)

        diff = w_new - w_init
        assert jnp.abs(jnp.mean(diff)) > 0
        log.info("grads is present")

    def trans_output_shape(self):
        out = self.transformer(self.x_test)

        assert out.shape == self.x_test.shape
        log.info("transformer output shape correct")

    def trans_grad(self):
        out_1 = self.transformer(self.x_test)

        optimizer = nnx.Optimizer(self.transformer, optax.adam(0.1), wrt=nnx.Param)

        def loss_fn(model, x):
            out = model(x)
            return jnp.sum(out)

        loss, grads = nnx.value_and_grad(loss_fn)(self.transformer, self.x_test)

        # get all the gradients - gradients are a tree
        leaves = jax.tree_util.tree_leaves(grads)
        for leaf in leaves:
            # check that at least each parameter recieves some gradient
            mini_grad = jnp.any(leaf != 0)
            assert mini_grad

        optimizer.update(self.transformer, grads)
        out_2 = self.transformer(self.x_test)

        diff = out_2 - out_1
        assert jnp.abs(jnp.mean(diff)) > 0
        log.info("grads is present")
