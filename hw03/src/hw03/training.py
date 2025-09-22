import jax.numpy as jnp
import numpy as np
import optax
import structlog
from flax import nnx
from tqdm import trange

from .config import TrainingSettings
from .data import Data_MNIST
from .model1 import Classifier2

log = structlog.get_logger()


def test_accuracy(
    model: Classifier2, data: Data_MNIST, batch_size: int, validation_set: bool = True
):
    correct = 0
    total = 0
    if validation_set:
        x_np, y_np = data.get_validation()
    else:
        x_np, y_np = data.get_test()

    n = len(y_np)

    for i in range(0, n, batch_size):
        x, y = (
            jnp.asarray(x_np[i : i + batch_size]),
            jnp.asarray(y_np[i : i + batch_size]),
        )
        fxy = model(x, train=False)
        pred = jnp.argmax(fxy, axis=1)
        log.debug("preditions", pred=pred)
        equal = pred == y
        log.debug("equals", equal=equal)
        total += len(equal)
        correct += jnp.sum(equal)

    accuracy_percent = correct / total
    log.info("total correct", correct=correct)
    log.info("total tested", total=total)
    log.info("accurary", accuracy=accuracy_percent)

    return accuracy_percent


@nnx.jit
def train_step_mnnist(
    model: Classifier2,
    optimizer: nnx.Optimizer,
    x: jnp.ndarray,
    y: jnp.ndarray,
):
    """perform on training step"""

    def loss_function(model: Classifier2):
        fxy = model(x, True)
        # calculate loss using cross entropy
        loss_ce = optax.softmax_cross_entropy_with_integer_labels(fxy, y)
        loss_ce = jnp.mean(loss_ce)
        l2_loss = model.l2loss()
        return loss_ce + l2_loss

    loss, grads = nnx.value_and_grad(loss_function)(model)
    optimizer.update(model, grads)
    return loss


def train_mnist(
    model: Classifier2,
    optimizer: nnx.Optimizer,
    data: Data_MNIST,
    settings: TrainingSettings,
    np_rng: np.random.Generator,
) -> None:
    """train with grad descend"""
    log.info("started training")
    bar = trange(settings.num_iters)
    for i in bar:
        x_np, y_np = data.get_batch(np_rng, settings.batch_size)
        log.debug("y_np", y_np=y_np)
        x, y = jnp.asarray(x_np), jnp.asarray(y_np)
        log.debug("y here", y=y)
        # x is (batch_size, 28, 28, 1)

        loss = train_step_mnnist(model, optimizer, x, y)

        bar.set_description(f"loss at {i} => {loss:.6f}")
        bar.refresh()
    log.info("done training")
