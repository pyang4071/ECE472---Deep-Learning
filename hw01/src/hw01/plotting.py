import matplotlib
import matplotlib.pyplot as plt
import jax.numpy as jnp
import numpy as np
import structlog

from .config import PlottingSettings
from .data import Data
from .model import NNXModel

log = structlog.get_logger()

font = {
    # "family": "Adobe Caslon Pro",
    "size": 10,
}

matplotlib.style.use("classic")
matplotlib.rc("font", **font)


def plot_fit(
    model: NNXModel,
    data: Data,
    settings: PlottingSettings,
):
    """Plots the linear fit and saves it to a file."""
    log.info("Plotting fit")
    fig, ax = plt.subplots(1, 1, figsize=settings.figsize, dpi=settings.dpi)

    ax.set_title("Linear fit")
    ax.set_xlabel("x")
    ax.set_ylim(np.amin(data.y) * 1.5, np.amax(data.y) * 1.5)
    h = ax.set_ylabel("y", labelpad=10)
    h.set_rotation(0)

    xs = np.linspace(0, 1, 50)
    xs = xs[:, np.newaxis]
    ax.plot(
        xs, np.squeeze(model(jnp.asarray(xs))), "-", np.squeeze(data.x), data.y, "o"
    )

    plt.tight_layout()

    settings.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = settings.output_dir / "hw01_sine.pdf"
    plt.savefig(output_path)
    log.info("Saved plot", path=str(output_path))


def plot_basis(
    model: NNXModel,
    settings: PlottingSettings,
    num_basis: int,
):
    """plot the basis functions"""
    log.info("Plotting basis functions")
    fig, ax = plt.subplots(1, 1, figsize=settings.figsize, dpi=settings.dpi)

    ax.set_title("Basis Functions")
    ax.set_xlabel("x")
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(0, 1.5)
    h = ax.set_ylabel("y", labelpad=10)
    h.set_rotation(0)

    x = np.linspace(-0.5, 1.5, 100)

    params = model.model_params
    for i in range(num_basis):
        u = params.means[i]
        sigma = params.variances[i]
        y = np.exp((-1 * ((x - u) ** 2)) / (sigma**2))
        ax.plot(x, y)

    plt.tight_layout()
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = settings.output_dir / "hw01_basis.pdf"
    plt.savefig(output_path)
    log.info("Saved plot", path=str(output_path))
