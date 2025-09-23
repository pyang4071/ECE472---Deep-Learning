import jax
import numpy as np
import optax
import structlog
from flax import nnx

from .config import load_settings
from .data import Data_MNIST
from .logging import configure_logging
from .model1 import Classifier2
from .training import train_mnist, test_accuracy


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

    data = Data_MNIST(
        rng=np_rng,
        split_ratio=settings.data.split_ratio,
    )
    log.info("downloaded data")

    model = Classifier2(
        input_depth=1,
        layer_depths=[32, 64],
        layer_kernel_sizes=[[3, 3], [3, 3]],
        strides=[1, 1],
        num_classes=10,
        dropout_rate=0.1,
        l2reg=settings.training.l2reg,
        rngs=nnx.Rngs(model_key),
        shape=[28, 28],
    )

    schedule = optax.cosine_decay_schedule(
        init_value=settings.training.learning_rate,
        decay_steps=settings.training.num_iters,
    )

    optimizer = nnx.Optimizer(model, optax.adam(schedule), wrt=nnx.Param)

    log.info("training")
    train_mnist(model, optimizer, data, settings.training, np_rng)

    log.info("testing on validation set")
    accuracy = test_accuracy(
        model=model,
        data=data,
        batch_size=settings.training.batch_size,
        validation_set=True,
    )

    if accuracy > 0.955:
        log.info("testing on test set")
        final_test_accuracy = test_accuracy(
            model=model,
            data=data,
            batch_size=settings.training.batch_size,
            validation_set=False,
        )
        log.info("Final test result: ", accuracy=final_test_accuracy)
    else:
        log.info("Fail on validation_set. Tune hyperparameters")
