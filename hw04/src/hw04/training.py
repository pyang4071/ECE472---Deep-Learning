import jax.numpy as jnp
import numpy as np
import optax
import structlog
from flax import nnx
from tqdm import trange

from .config import TrainingSettings
from .data import Data_CIFAR
from .model1 import Classifier

log = structlog.get_logger()


def test_accuracy(
    model: Classifier, data: Data_CIFAR, batch_size: int, validation_set: bool = True
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
        fxy = model(x)
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


def data_augment(
    x: np.ndarray,
    settings: TrainingSettings,
    np_rng: np.random.Generator,
):
    """randomly data augment"""
    batch_size = x.shape[0]
    # flip over y
    flip_mask = np_rng.random(batch_size) < settings.flip_ratio
    for i in range(batch_size):
        if flip_mask[i]:
            x[i] = np.flip(x[i], axis=1)

    # add noise
    noise = np_rng.normal(loc=0.0, scale=settings.noise_std, size=x.shape)
    x_noised = x + noise
    x_noised = np.clip(x_noised, 0.0, 1.0)
    return x_noised


@nnx.jit
def train_step_CIFAR(
    model: Classifier,
    optimizer: nnx.Optimizer,
    x: jnp.ndarray,
    y: jnp.ndarray,
):
    """perform on training step"""

    def loss_function(model: Classifier):
        fxy = model(x)
        # calculate loss using cross entropy
        loss_ce = optax.softmax_cross_entropy_with_integer_labels(fxy, y)
        loss_ce = jnp.mean(loss_ce)
        l2_loss = model.l2loss()
        return loss_ce + l2_loss

    loss, grads = nnx.value_and_grad(loss_function)(model)
    optimizer.update(model, grads)
    return loss


def train_CIFAR(
    model: Classifier,
    optimizer: nnx.Optimizer,
    data: Data_CIFAR,
    settings: TrainingSettings,
    np_rng: np.random.Generator,
) -> None:
    """train with grad descend"""
    log.info("started training")
    bar = trange(settings.num_iters)
    for i in bar:
        x_np, y_np = data.get_batch(np_rng, settings.batch_size)
        x_np = data_augment(x_np, settings, np_rng)

        log.debug("y_np", y_np=y_np)
        x, y = jnp.asarray(x_np), jnp.asarray(y_np)

        log.debug("y here", y=y)
        # x is (batch_size, 32, 32, 3)

        loss = train_step_CIFAR(model, optimizer, x, y)

        bar.set_description(f"loss at {i} => {loss:.6f}")
        bar.refresh()
    log.info("done training")
