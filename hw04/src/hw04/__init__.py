import jax
import numpy as np
import optax
import orbax.checkpoint as ocp
from pathlib import Path
import structlog
from flax import nnx

from .config import load_settings
from .data import Data_CIFAR
from .logging import configure_logging
from .model1 import Classifier
from .training import train_CIFAR, test_accuracy

"""
We will define the state of the art as having 95.5 percent accuracy
"""


def main() -> None:
    """command line interface entry"""
    settings = load_settings()
    configure_logging()
    log = structlog.get_logger()
    log.info("Settings loaded", settings=settings.model_dump())

    # JAX PRNG
    key = jax.random.PRNGKey(settings.random_seed)
    data_key, model_key = jax.random.split(key)
    np_rng = np.random.default_rng(np.array(data_key))

    data_10 = Data_CIFAR(
        rng=np_rng,
        split_ratio=settings.data.split_ratio,
        CIFAR10=True,
    )
    log.info("downloaded CIFAR10 data")

    model = Classifier(
        input_depth=settings.model.input_depth,
        layer_depths=settings.model.layer_depths,
        layer_kernel_size=settings.model.kernel,
        strides=settings.model.stride,
        num_classes=settings.model.num_classes,
        l2reg=settings.model.l2reg,
        rngs=nnx.Rngs(model_key),
        input_shape=settings.model.data_shape,
        num_groups=settings.model.num_groups,
    )

    schedule = optax.cosine_decay_schedule(
        init_value=settings.training.learning_rate,
        decay_steps=settings.training.num_iters,
    )

    optimizer = nnx.Optimizer(model, optax.adam(schedule), wrt=nnx.Param)

    log.info("training")
    train_CIFAR(model, optimizer, data_10, settings.training, np_rng)

    log.info("testing on validation set")
    test_accuracy(
        model=model,
        data=data_10,
        batch_size=settings.training.batch_size,
        validation_set=True,
    )

    # checkpoint this
    ckpt_dir = ocp.test_utils.erase_and_create_empty("/tmp/cifar10-150000/")
    _, state = nnx.split(model)
    # nnx.display(state)

    checkpointer = ocp.StandardCheckpointer()
    checkpointer.save(ckpt_dir / "cifar10-resnet20", state)
    checkpointer.wait_until_finished()
    log.info("saved model")


def test_on_test_set():
    settings = load_settings()
    configure_logging()
    log = structlog.get_logger()
    log.info("Settings loaded", settings=settings.model_dump())

    # JAX PRNG
    key = jax.random.PRNGKey(settings.random_seed)
    data_key, model_key = jax.random.split(key)
    np_rng = np.random.default_rng(np.array(data_key))

    data_10 = Data_CIFAR(
        rng=np_rng,
        split_ratio=settings.data.split_ratio,
        CIFAR10=True,
    )
    log.info("downloaded CIFAR10 data")

    model = Classifier(
        input_depth=settings.model.input_depth,
        layer_depths=settings.model.layer_depths,
        layer_kernel_size=settings.model.kernel,
        strides=settings.model.stride,
        num_classes=settings.model.num_classes,
        l2reg=settings.model.l2reg,
        rngs=nnx.Rngs(model_key),
        input_shape=settings.model.data_shape,
        num_groups=settings.model.num_groups,
    )

    # recreate model
    ckpt_dir = Path("/tmp/cifar10-pooling/")
    checkpointer = ocp.StandardCheckpointer()
    graphdef, state = nnx.split(model)
    state_restored = checkpointer.restore(ckpt_dir / "cifar10-resnet20", state)
    model = nnx.merge(graphdef, state_restored)

    log.info("testing on test set")
    test_accuracy(
        model=model,
        data=data_10,
        batch_size=settings.training.batch_size,
        validation_set=False,
    )
