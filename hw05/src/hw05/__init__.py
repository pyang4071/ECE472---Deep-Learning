import jax
import numpy as np
import optax
import orbax.checkpoint as ocp
from pathlib import Path
import structlog
from flax import nnx

from .config import load_settings
from .data import Data
from .logging import configure_logging
from .model1 import Classifier
from .training import train, test_accuracy


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

    data = Data(
        rng=np_rng,
        num_splits=settings.training.num_fold,
        token_length=settings.model.token_length,
        max_token=settings.data.max_token,
    )

    acc = np.zeros(settings.training.num_fold)
    for fold in range(settings.training.num_fold):
        model = Classifier(
            rngs=nnx.Rngs(params=model_key),
            max_token=settings.data.max_token,
            embed_depth=settings.model.embed_depth,
            num_outputs=settings.model.num_classes,
            num_layers=settings.model.num_layers,
            hidden_layer_width=settings.model.hidden_layer_width,
            l2reg=settings.training.l2reg,
        )

        schedule = optax.cosine_decay_schedule(
            init_value=settings.training.learning_rate,
            decay_steps=settings.training.num_iters,
        )

        optimizer = nnx.Optimizer(model, optax.adam(schedule), wrt=nnx.Param)

        acc[fold] = train(model, optimizer, data, settings.training, np_rng, fold)

    log.info("mean val accuracy:", val=np.mean(acc))

    ckpt_dir = ocp.test_utils.erase_and_create_empty(settings.save.checkpoint_save)
    _, state = nnx.split(model)

    checkpointer = ocp.StandardCheckpointer()
    checkpointer.save(ckpt_dir / "ag_news", state)
    checkpointer.wait_until_finished()
    log.info(f"saved model at {settings.save.checkpoint_save}")


def test_on_test_set():
    settings = load_settings()
    configure_logging()
    log = structlog.get_logger()
    log.info("Settings loaded", settings=settings.model_dump())

    # JAX PRNG
    key = jax.random.PRNGKey(settings.random_seed)
    data_key, model_key = jax.random.split(key)
    np_rng = np.random.default_rng(np.array(data_key))

    data = Data(
        rng=np_rng,
        num_splits=settings.training.num_fold,
        token_length=settings.model.token_length,
        max_token=settings.data.max_token,
    )

    model = Classifier(
        rngs=nnx.Rngs(params=model_key),
        max_token=settings.data.max_token,
        embed_depth=settings.model.embed_depth,
        num_outputs=settings.model.num_classes,
        num_layers=settings.model.num_layers,
        hidden_layer_width=settings.model.hidden_layer_width,
        l2reg=settings.training.l2reg,
    )

    # recreate model
    ckpt_dir = Path(settings.save.checkpoint_load)
    checkpointer = ocp.StandardCheckpointer()
    graphdef, state = nnx.split(model)
    state_restored = checkpointer.restore(ckpt_dir / "ag_news", state)
    model = nnx.merge(graphdef, state_restored)

    log.info("testing on test set")
    test = test_accuracy(
        model, data, settings.training.batch_size, validation_set=False
    )
    log.info("Final test accuracy", accuracy=test)
