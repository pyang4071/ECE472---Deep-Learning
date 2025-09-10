import jax.numpy as jnp
import numpy as np
import structlog
from flax import nnx
from tqdm import trange

from .config import TrainingSettings
from .data import Data
from .model import  NNXModel

log = structlog.get_logger()


@nnx.jit
def train_step_gs(
    model: NNXModel,
    optimizer: nnx.Optimizer,
    x: jnp.ndarray,
    y: jnp.ndarray
):
    '''perform a single training step'''
    def loss_function(model: NNXModel):
        y_hat = model(x)
        return 0.5 * jnp.mean((y_hat - y) ** 2) # mean to average the matrix to get avg loss 
    loss, grads = nnx.value_and_grad(loss_function)(model)
    optimizer.update(model, grads) # update parameters
    return loss # to record

def train_gs(
    model: NNXModel,
    optimizer: nnx.Optimizer,
    data: Data,
    settings: TrainingSettings,
    np_rng: np.random.Generator,
):
    '''train the model with stochastic gradient descend'''
    log.info("stating training")
    bar = trange(settings.num_iters)
    for i in bar:
        x_np, y_np = data.get_batch(np_rng, settings.batch_size)
        x, y = jnp.asarray(x_np), jnp.asarray(y_np) # convert to jax
        loss = train_step_gs(model, optimizer, x, y) # do one iteration

        bar.set_description(f"Loss at {i} => {loss:.6f}") # update progress with loss
        bar.refresh()
    log.info("finished training")



