import jax
import numpy as np
import optax
import orbax.checkpoint as ocp
from pathlib import Path
import structlog
from flax import nnx

from .config import load_settings
from .data import Data_Spiral, Data_Activations
from .logging import configure_logging
from .model import NNXSpiralModel, SparseEncoder
from .plotting import plot_fit, plot_features
from .training import train_spiral, train_encoder


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
        key=model_key,
        in_dim=settings.data.num_features,
        layer_dim=settings.model.layer_dim,
    )
    log.info("created model")

    schedule = optax.cosine_decay_schedule(
        init_value=settings.training.learning_rate,
        decay_steps=settings.training.num_iters,
    )

    optimizer = nnx.Optimizer(model, optax.adam(schedule), wrt=nnx.Param)

    train_spiral(model, optimizer, data, settings.training, np_rng)
    log.info("finish training")

    plot_fit(model, data, settings.plotting)
    log.info("created plot")

    ckpt_dir = ocp.test_utils.erase_and_create_empty(settings.save.checkpoint_save_mlp)
    _, state = nnx.split(model)
    # nnx.display(state)

    checkpointer = ocp.StandardCheckpointer()
    checkpointer.save(ckpt_dir / "spiral_sparse", state)
    checkpointer.wait_until_finished()
    log.info("saved model")


def train_sparse():
    settings = load_settings()
    configure_logging()
    log = structlog.get_logger()
    log.info("Settings loaded", settings=settings.model_dump())

    # JAX PRNG
    key = jax.random.PRNGKey(settings.random_seed + 1)
    data_key, classifer_key, encoder_key = jax.random.split(key, 3)
    np_rng = np.random.default_rng(np.array(data_key))

    classifier = NNXSpiralModel(
        key=classifer_key,
        in_dim=settings.data.num_features,
        layer_dim=settings.model.layer_dim,
    )

    # recreate classifier
    ckpt_dir = Path(settings.save.checkpoint_load_mlp)
    checkpointer = ocp.StandardCheckpointer()
    graphdef, state = nnx.split(classifier)
    state_restored = checkpointer.restore(ckpt_dir / "spiral_sparse", state)
    classifier = nnx.merge(graphdef, state_restored)

    data_act = Data_Activations(
        rng=np_rng,
        model=classifier,
        num=settings.data.enc_num_data,
        act_dim=settings.model.layer_dim[-1],
    )

    sparse_enc = SparseEncoder(
        key=encoder_key,
        in_dim=settings.model.layer_dim[-1],
        hidden_dim=settings.model.enc_hidden_dim,
        norm_val=settings.model.enc_norm_val,
    )

    schedule = optax.piecewise_interpolate_schedule(
        interpolate_type="linear",
        init_value=settings.training.enc_lr,
        boundaries_and_scales={
            0.8 * settings.training.enc_num_iters: 1,
            settings.training.enc_num_iters: 0,
        },
    )

    optimizer = nnx.Optimizer(
        sparse_enc,
        optax.chain(
            optax.clip_by_global_norm(1),
            optax.adam(schedule, b1=0.9, b2=0.999),
        ),
        wrt=nnx.Param,
    )

    train_encoder(sparse_enc, optimizer, data_act, settings.training, np_rng)

    ckpt_dir = ocp.test_utils.erase_and_create_empty(settings.save.checkpoint_save_enc)
    _, state = nnx.split(sparse_enc)
    # nnx.display(state)

    checkpointer = ocp.StandardCheckpointer()
    checkpointer.save(ckpt_dir / "enc_sparse", state)
    checkpointer.wait_until_finished()
    log.info("saved model", loc=f"{ckpt_dir}/enc_sparse")


def features():
    settings = load_settings()
    configure_logging()
    log = structlog.get_logger()
    log.info("Settings loaded", settings=settings.model_dump())

    # JAX PRNG
    key = jax.random.PRNGKey(settings.random_seed + 1)
    data_key, classifer_key, encoder_key = jax.random.split(key, 3)

    classifier = NNXSpiralModel(
        key=classifer_key,
        in_dim=settings.data.num_features,
        layer_dim=settings.model.layer_dim,
    )

    # recreate classifier
    ckpt_dir = Path(settings.save.checkpoint_load_mlp)
    checkpointer = ocp.StandardCheckpointer()
    graphdef, state = nnx.split(classifier)
    state_restored = checkpointer.restore(ckpt_dir / "spiral_sparse", state)
    classifier = nnx.merge(graphdef, state_restored)

    log.info("recreated classifer")

    # recreate sparse encoder
    sparse_enc = SparseEncoder(
        key=encoder_key,
        in_dim=settings.model.layer_dim[-1],
        hidden_dim=settings.model.enc_hidden_dim,
        norm_val=settings.model.enc_norm_val,
    )

    ckpt_dir = Path(settings.save.checkpoint_load_enc)
    checkpointer = ocp.StandardCheckpointer()
    graphdef, state = nnx.split(sparse_enc)
    state_restored = checkpointer.restore(ckpt_dir / "enc_sparse", state)
    sparse_enc = nnx.merge(graphdef, state_restored)

    log.info("recreate sparse encoder")

    plot_features(classifier, sparse_enc, settings.plotting)

    plot_features(classifier, sparse_enc, settings.plotting, True)

    log.info("finished all plotting")


"""
The sparse layer must be large enough such the features are distinctly separated. If not, the features would over lay themselves on each other which may make anaylsis difficult.
In the case of the psiral, we have found features that each correspond to a region in space. This suggest that the classifier take in the data and determine the location where on the grid it is before making a guess.
We notice the features are activated in a spiral like fashion which is similiar to the spirals seen in the dataset. Thus, it must mean that the model is breaking the space into spirals. 

"""
