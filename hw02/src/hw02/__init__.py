import jax
import numpy as np
import optax
import structlog
from flax import nnx

from .config import load_settings
from .data import Data_Spiral
from .logging import configure_logging
from .model import NNXSpiralModel
from .plotting import plot_fit
from .training import train_spiral


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

    data = Data_Spiral(
        rng=np_rng,
        num_features=settings.data.num_features,
        num_samples=settings.data.num_samples,
        sigma=settings.data.sigma_noise,
    )
    log.info("created data")

    model = NNXSpiralModel(
        rngs=nnx.Rngs(params=model_key),
        num_inputs=settings.data.num_features,
        num_outputs=1,
        num_layers=settings.training.num_layers,
        hidden_layer_width=settings.training.hidden_layer_width,
    )
    log.info("created model")

    optimizer = nnx.Optimizer(
        model, optax.adam(settings.training.learning_rate), wrt=nnx.Param
    )

    train_spiral(model, optimizer, data, settings.training, np_rng)
    log.info("finish training")

    plot_fit(model, data, settings.plotting)
    log.info("created plot")
