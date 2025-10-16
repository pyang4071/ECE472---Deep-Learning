import jax.numpy as jnp
import numpy as np
import optax
import structlog
from flax import nnx
from tqdm import trange

from .config import TrainingSettings
from .data import Data
from .model1 import Classifier

log = structlog.get_logger()


def test_accuracy(
    model: Classifier,
    data: Data,
    batch_size: int,
    validation_set: bool = True,
    val_set: int = 0,
):
    correct = 0
    total = 0
    if validation_set:
        x_np, y_np = data.get_validation(val_set)
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

    return accuracy_percent


@nnx.jit
def train_step(
    model: Classifier,
    optimizer: nnx.Optimizer,
    x: jnp.ndarray,
    y: jnp.ndarray,
):
    """perform on training step"""

    def loss_function(model: Classifier):
        fxy = model(x)
        # jax.debug.print("fxy = {}", fxy)
        # jax.debug.print("Predictions: {pred}", pred=jnp.argmax(fxy, axis=1))
        # jax.debug.print("True labels: {true}", true=y)
        # calculate loss using cross entropy
        loss_ce = optax.softmax_cross_entropy_with_integer_labels(fxy, y)
        loss_ce = jnp.mean(loss_ce)
        l2_loss = model.l2loss()
        return loss_ce + l2_loss

    loss, grads = nnx.value_and_grad(loss_function)(model)
    optimizer.update(model, grads)
    return loss


def train(
    model: Classifier,
    optimizer: nnx.Optimizer,
    data: Data,
    settings: TrainingSettings,
    np_rng: np.random.Generator,
    fold: int,
) -> None:
    """train with grad descend"""
    log.info(f"started training fold {fold}")

    bar = trange(settings.num_iters)
    for i in bar:
        x_np, y_np = data.get_batch(np_rng, settings.batch_size, fold)

        x, y = jnp.asarray(x_np), jnp.asarray(y_np)

        loss = train_step(model, optimizer, x, y)

        bar.set_description(f"loss at {i} => {loss:.6f}")
        bar.refresh()

    valid_acc = test_accuracy(model, data, settings.batch_size, True, fold)
    log.info(f"accuracy at fold {fold}: {valid_acc:.6f}")

    return valid_acc
