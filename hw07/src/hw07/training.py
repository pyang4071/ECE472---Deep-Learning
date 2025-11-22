import jax.numpy as jnp
import numpy as np
import structlog
from flax import nnx
from tqdm import trange

from .config import TrainingSettings
from .data import Data_Spiral
from .model import NNXSpiralModel, SparseEncoder

log = structlog.get_logger()


@nnx.jit
def train_step_spiral(
    model: NNXSpiralModel,
    optimizer: nnx.Optimizer,
    x: jnp.ndarray,
    t: jnp.ndarray,
    epsilon: float,
    l2loss: float,
):
    """perform one training step"""

    def loss_function(model: NNXSpiralModel):
        prob = model(x)

        # Cross entropy loss
        prob = jnp.clip(prob, epsilon, 1 - epsilon)  # to avoid infy loss
        ce_loss = -((t * jnp.log(prob)) + ((1 - t) * (jnp.log(1 - prob))))
        ce_loss = jnp.mean(ce_loss)

        # l2loss
        l2_loss = l2loss * model.l2loss()

        return ce_loss + l2_loss

    loss, grads = nnx.value_and_grad(loss_function)(model)
    optimizer.update(model, grads)

    return loss


def train_spiral(
    model: NNXSpiralModel,
    optimizer: nnx.Optimizer,
    data: Data_Spiral,
    settings: TrainingSettings,
    np_rng: np.random.Generator,
) -> None:
    """train with stochastic grad descend"""
    log.info("Start training")
    bar = trange(settings.num_iters)
    for i in bar:
        x_np, t_np = data.get_batch(np_rng, settings.batch_size)
        x, t = jnp.asarray(x_np), jnp.asarray(t_np)
        # x is (num_samples, features)
        # t is (num_samples,)
        t = t.reshape(-1, 1)  # t is (num_samples, 1)

        loss = train_step_spiral(
            model, optimizer, x, t, settings.log_clip, settings.l2_loss
        )

        bar.set_description(f"Loss at {i} => {loss:.6f}")
        bar.refresh()
    log.info("Training Finished")


@nnx.jit
def train_step_encoder(
    model: SparseEncoder, optimizer: nnx.Optimizer, x: jnp.ndarray, lam: float
):
    """one step of training"""

    def loss_function(model: SparseEncoder):
        fx, x_hat, dec_w = model(x)
        # X is the given data sample
        # |X| = num of samples => first term is mean of l2 norms
        loss_t1 = jnp.mean(jnp.square(jnp.linalg.norm(x - x_hat, axis=1)))

        w_col_l2norm = jnp.linalg.norm(dec_w, axis=1)
        # shape (embed_dim, ) aka norm over each decoder row
        mag_fx = jnp.abs(fx)

        inner = mag_fx * w_col_l2norm
        # each column is an element in the sum (property per feature)

        # sum each column to each other for the summation
        loss_t2 = jnp.sum(inner, axis=1)
        # should be a column vector of the loss per sample

        # vector of loss per sample
        loss_t2 = jnp.mean(loss_t2)
        # single scaler - mean over the batch

        loss = loss_t1 + loss_t2
        return loss

    loss, grads = nnx.value_and_grad(loss_function)(model)
    optimizer.update(model, grads)

    return loss


def train_encoder(
    model: SparseEncoder,
    optimizer: nnx.Optimizer,
    data: Data_Spiral,
    settings: TrainingSettings,
    np_rng: np.random.Generator,
) -> None:
    """train sparse encoder via grad descend"""
    log.info("start sparse encoder training")
    bar = trange(settings.enc_num_iters)
    for i in bar:
        x_np = data.get_batch(np_rng, settings.enc_batch_size)
        x = jnp.asarray(x_np)
        lam = determine_lam_enc(i, settings.enc_num_iters, settings.enc_lam)
        loss = train_step_encoder(model, optimizer, x, lam)
        bar.set_description(f"Loss at {i} => {loss:.6f}")
        bar.refresh()

    log.info("encoder training finished")


def determine_lam_enc(cur_iter: float, total_iter: int, final_lam: int):
    increment_steps = 0.05 * total_iter
    slope = final_lam / increment_steps
    if cur_iter < increment_steps:
        lam = slope * cur_iter
    else:
        lam = final_lam

    return lam
