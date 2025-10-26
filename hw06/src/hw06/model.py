import jax
import jax.numpy as jnp
from flax import nnx
import structlog


log = structlog.get_logger()


class Multihead_attention(nnx.Module):
    def __init__(self, key, num_head: int, embed_depth: int):
        self.keys = jax.random.split(key, 4)
        # w_i is shape (embed_depth, dk)
        # dk = embed_depth/num_head
        # dv = dk
        assert embed_depth % num_head == 0

        self.num_head = num_head
        self.dk = embed_depth // num_head
        dv = self.dk
        self.w_q = nnx.Param(
            jax.random.normal(self.keys[0], (num_head, embed_depth, self.dk))
            * jnp.sqrt(2 / num_head)
        )
        self.w_k = nnx.Param(
            jax.random.normal(self.keys[1], (num_head, embed_depth, self.dk))
            * jnp.sqrt(2 / num_head)
        )
        self.w_v = nnx.Param(
            jax.random.normal(self.keys[2], (num_head, embed_depth, dv))
            * jnp.sqrt(2 / num_head)
        )

        self.w_o = nnx.Param(
            jax.random.normal(self.keys[3], (num_head * self.dk, embed_depth))
            * jnp.sqrt(2 / (num_head * self.dk))
        )

        self.scale = jnp.sqrt(1 / self.dk)

    def __call__(self, x_q, x_k, x_v, masking: bool, ret_weights: bool = False):
        # x_q is (batch, seq_length_q, embed_depth)
        # x_k=x_v is (batch, seq_length_k, embed_depth)
        batch, seq_length_q, _ = x_q.shape
        _, seq_length_k, _ = x_k.shape

        if masking:
            mask = jnp.tril(jnp.ones([seq_length_q, seq_length_k]))
            causal_mask = (1 - mask) * -1e9
            # 0 means pass and -1e9 is blocking (masking)

        atten_weights = jnp.zeros([batch, self.num_head, seq_length_q, seq_length_k])
        outputs = jnp.zeros([batch, seq_length_q, self.num_head, self.dk])
        for i in range(self.num_head):
            w_qi = self.w_q[i, :, :]
            w_ki = self.w_k[i, :, :]
            w_vi = self.w_v[i, :, :]
            # shape is (embed_depth, dk)

            q_i = x_q @ w_qi
            k_i = x_k @ w_ki
            v_i = x_v @ w_vi
            # shape (batch, seq_length, dk)

            head_out = jnp.zeros([batch, seq_length_q, self.dk])
            for j in range(batch):
                # iterate over each batch item
                # idk how to do the matmul at once...
                q_i_b = q_i[j, :, :]
                k_i_b = k_i[j, :, :]
                v_i_b = v_i[j, :, :]
                # shape (seq_length, dk)

                inner = q_i_b @ k_i_b.T / self.scale
                if masking:
                    inner = inner + causal_mask
                # shape is (seq_length_q, seq_length_k)

                prob = jax.nn.softmax(inner)
                # shape is (seq_length_q, seq_length_k)
                atten_weights = atten_weights.at[j, i, :, :].set(prob)

                atten = prob @ v_i_b
                # shape = (seq_length_q, dk)

                head_out = head_out.at[j, :, :].set(atten)

            outputs = outputs.at[:, :, i, :].set(head_out)

        # concat
        outputs = outputs.reshape(batch, seq_length_q, self.num_head * self.dk)

        mha = outputs @ self.w_o
        # shape is (batch, seq_length_q, embed_depth)

        if ret_weights:
            return mha, atten_weights

        return mha


class Feedforward(nnx.Module):
    def __init__(self, key, embed_depth: int, ff_dim: int):
        self.keys = jax.random.split(key, 2)
        self.w_1 = nnx.Param(jax.random.normal(self.keys[0], (embed_depth, ff_dim)))
        self.b_1 = nnx.Param(jnp.zeros((1, 1, ff_dim)))

        self.w_2 = nnx.Param(jax.random.normal(self.keys[1], (ff_dim, embed_depth)))
        self.b_2 = nnx.Param(jnp.zeros((1, 1, embed_depth)))
        self.activation = jax.nn.relu

    def __call__(self, x):
        fx = x @ self.w_1 + self.b_1
        # shape (batch, seq_length, ff_dim)

        fx = self.activation(fx)

        fx = fx @ self.w_2 + self.b_2
        # shape (batch, seq_length, embed_depth)

        return x + fx


class Transformer(nnx.Module):
    def __init__(self, key, num_head: int, embed_depth: int, ff_dim: int):
        self.keys = jax.random.split(key, 5)
        self.multihead_attention_enc = Multihead_attention(
            key=self.keys[0],
            num_head=num_head,
            embed_depth=embed_depth,
        )
        self.ff_1 = Feedforward(
            key=self.keys[1],
            embed_depth=embed_depth,
            ff_dim=ff_dim,
        )
        self.multihead_attention_dec = Multihead_attention(
            key=self.keys[2],
            num_head=num_head,
            embed_depth=embed_depth,
        )
        self.multihead_attention_cross = Multihead_attention(
            key=self.keys[3],
            num_head=num_head,
            embed_depth=embed_depth,
        )
        self.ff_2 = Feedforward(
            key=self.keys[4],
            embed_depth=embed_depth,
            ff_dim=ff_dim,
        )

    def layer_norm(self, x, eps: float = 1e-5):
        mean = jnp.mean(x, axis=-1, keepdims=True)
        var = jnp.var(x, axis=-1, keepdims=True)
        normed = (x - mean) / jnp.sqrt(var + eps)
        return normed

    def encoder(self, x):
        x = x + self.multihead_attention_enc(x, x, x, False)
        x = self.layer_norm(x)
        # x is (batch, seq_length_q, embed_depth)

        x = self.ff_1(x)
        x = self.layer_norm(x)
        return x

    def decoder(self, x):
        enc_out = x
        x = self.multihead_attention_dec(x, x, x, True)
        # x is shape (batch, seq_dec, embed_depth)

        x = self.layer_norm(x)
        x = x + self.multihead_attention_cross(x, enc_out, enc_out, False)
        # shape is (batch, seq_dec, embed_depth)

        x = self.layer_norm(x)
        x = x + self.ff_2(x)
        # shape is (batch, seq_dec, embed_depth)

        x = self.layer_norm(x)
        return x

    def __call__(self, x):
        # x is shape (batch, seq_length, embed_depth)

        # Encoder section
        enc_output = self.encoder(x)
        # shape x is (batch, seq_length_enc, embed_depth)

        # Decoder section
        dec_out = self.decoder(enc_output)

        return dec_out
