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
from .training import train_CIFAR, test_accuracy, test_top5_accuracy


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

    if settings.model.cifar10:
        num_class = 10
    else:
        num_class = 100

    data_10 = Data_CIFAR(
        rng=np_rng,
        split_ratio=settings.data.split_ratio,
        CIFAR10=settings.model.cifar10,
    )
    log.info("finish data preprocessing")

    model = Classifier(
        input_depth=settings.model.input_depth,
        layer_depths=settings.model.layer_depths,
        layer_kernel_size=settings.model.kernel,
        strides=settings.model.stride,
        num_classes=num_class,
        l2reg=settings.model.l2reg,
        rngs=nnx.Rngs(model_key),
        input_shape=settings.model.data_shape,
        num_groups=settings.model.num_groups,
    )

    schedule = optax.cosine_decay_schedule(
        init_value=settings.training.learning_rate,
        decay_steps=settings.training.num_iters,
    )

    optimizer = nnx.Optimizer(
        model,
        optax.sgd(learning_rate=schedule, momentum=settings.training.momentum),
        wrt=nnx.Param,
    )

    log.info("training")
    train_CIFAR(model, optimizer, data_10, settings.training, np_rng)

    log.info("testing on validation set")
    if settings.model.top_1:
        test_accuracy(
            model=model,
            data=data_10,
            batch_size=settings.training.batch_size,
            validation_set=True,
        )
    else:
        test_top5_accuracy(
            model=model,
            data=data_10,
            batch_size=settings.training.batch_size,
            validation_set=True,
        )

    # checkpoint this
    ckpt_dir = ocp.test_utils.erase_and_create_empty(settings.save.checkpoint_save)
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

    if settings.model.cifar10:
        num_class = 10
    else:
        num_class = 100

    data_10 = Data_CIFAR(
        rng=np_rng,
        split_ratio=settings.data.split_ratio,
        CIFAR10=settings.model.cifar10,
    )
    log.info("finish data preprocessing")

    model = Classifier(
        input_depth=settings.model.input_depth,
        layer_depths=settings.model.layer_depths,
        layer_kernel_size=settings.model.kernel,
        strides=settings.model.stride,
        num_classes=num_class,
        l2reg=settings.model.l2reg,
        rngs=nnx.Rngs(model_key),
        input_shape=settings.model.data_shape,
        num_groups=settings.model.num_groups,
    )

    # recreate model
    ckpt_dir = Path(settings.save.checkpoint_load)
    checkpointer = ocp.StandardCheckpointer()
    graphdef, state = nnx.split(model)
    state_restored = checkpointer.restore(ckpt_dir / "cifar10-resnet20", state)
    model = nnx.merge(graphdef, state_restored)

    log.info("testing on test set")
    if settings.model.top_1:
        test_accuracy(
            model=model,
            data=data_10,
            batch_size=settings.training.batch_size,
            validation_set=False,
        )
    else:
        test_top5_accuracy(
            model=model,
            data=data_10,
            batch_size=settings.training.batch_size,
            validation_set=False,
        )


"""
The model was first using vmap and scan to iterate over residual blocks but this forced a fixed input and output dimensions over each vmapped block. Layer sizes of 32 and 64 were test on small scale trials but each iteration took a long time for low accuracy on a cpu. Thus, the classifier model was altered to allow for changing shapes across each residual block. The layer sizes and stride were taken from the mnist homework leading to suboptimal performance. Thus, taking inspiration from Kaiming He et al paper on "Deep Residual Learning for Image Recgnition," the channel numbers and strides are modified to reflect the ResNet 20 architecture. Furthermore, each residual block are designed with a pre-activation and a identity activation. A 1x1 convolution is added to the shortcut only when the shape of the output changes from the input. This residual block architecture was used in all trials in both fixed size and varying size residual blocks. 

Different data augmentations were used. At first, we only adapted flipping before including spacial shifts and randomly added noise. Truth be told, these addded augmenentations did not improve the accuracy on the valid by a significant amount. 

The final result for Cifar10 uses the setup shown in the config.toml and it run for approximately 10-12 hours of continuous running with approximately 1.7-1.8 iterations per second. As such, deeper models were not possible to be adapted on this configuration. The Cifar100 was tested using the same architecture as Cifar10 and produced good results. A little too good that I feel like I did something wrong.


Final test accuracy top1 Cifar10 : 0.8600
Final test accuracy top1 Cifar100: 0.8619
Final test accuracy top5 Cifar100: 0.9938
"""
