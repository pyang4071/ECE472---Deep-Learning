import jax.numpy as jnp
import numpy as np
import structlog
from flax import nnx
from tqdm import trange

from .config import TrainingSettings
from .data import Data_Spiral
from .model import NNXSpiralModel

log = structlog.get_logger()


@nnx.jit
def train_step_spiral(
    model: NNXSpiralModel,
    optimizer: nnx.Optimizer,
    x: jnp.ndarray,
    t: jnp.ndarray,
    epsilon: float,
):
    """perform one training step"""

    def loss_function(model: NNXSpiralModel):
        fxy = model(x)
        fxy = jnp.clip(fxy, epsilon, 1 - epsilon)  # to avoid infy loss
        log.debug("loss function shape", fxy=fxy.shape)
        loss = -((t * jnp.log(fxy)) + ((1 - t) * (jnp.log(1 - fxy))))
        loss = jnp.mean(loss)
        return loss

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

        loss = train_step_spiral(model, optimizer, x, t, settings.log_clip)
        log.debug("loss", loss=loss)

        bar.set_description(f"Loss at {i} => {loss:.6f}")
        bar.refresh()
    log.info("Training Finished")
