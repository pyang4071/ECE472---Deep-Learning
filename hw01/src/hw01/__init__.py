import jax
import numpy as np
import optax
import structlog
from flax import nnx

from .config import load_settings
from .data import Data
from .logging import configure_logging
from .model import NNXModel
from .plotting import plot_fit, plot_basis
from .training import train_gs


def main() -> None:
    """CLI entry point."""
    settings = load_settings()
    configure_logging()
    log = structlog.get_logger()
    log.info("Settings loaded", settings=settings.model_dump())

    # JAX PRNG
    key = jax.random.PRNGKey(settings.random_seed)
    data_key, model_key = jax.random.split(key)
    np_rng = np.random.default_rng(np.array(data_key))
    log.info("Generated random keys.")

    data = Data(
        rng=np_rng,
        num_samples=settings.data.num_samples,
        sigma=settings.data.sigma_noise,
        bias=settings.data.bias,
    )

    log.info("created data")

    model = NNXModel(
        rngs=nnx.Rngs(params=model_key), num_basis=settings.training.num_basis
    )
    log.info("Generated model")

    optimizer = nnx.Optimizer(
        model, optax.adam(settings.training.learning_rate), wrt=nnx.Param
    )

    train_gs(model, optimizer, data, settings.training, np_rng)

    log.info("trained model")

    if settings.data.num_features == 1:
        plot_fit(model, data, settings.plotting)
        plot_basis(model, settings.plotting, settings.training.num_basis)
    else:
        log.info("Skipping plotting for multi-feature models.")
