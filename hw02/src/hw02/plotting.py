import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.inspection import DecisionBoundaryDisplay
import structlog

from .config import PlottingSettings
from .data import Data_Spiral
from .model import NNXSpiralModel

log = structlog.get_logger()

font = {
    # "family": "Adobe Caslon Pro",
    "size": 10,
}

matplotlib.style.use("classic")
matplotlib.rc("font", **font)


def plot_fit(
    model: NNXSpiralModel,
    data: Data_Spiral,
    settings: PlottingSettings,
):
    """Plots the linear fit and saves it to a file."""
    log.info("plotting")

    feature1, feature2 = np.meshgrid(
        np.linspace(
            data.x[:, 0].min() * 1.1, data.x[:, 0].max() * 1.1, settings.linspace
        ),
        np.linspace(
            data.x[:, 1].min() * 1.1, data.x[:, 1].max() * 1.1, settings.linspace
        ),
    )
    grid = np.vstack([feature1.ravel(), feature2.ravel()]).T
    log.debug("grid = {}", grid=grid)
    t_pred = model(grid)
    log.debug("feature1 shape", shape=feature1.shape)
    log.debug("t_pred shape", shape=t_pred.shape)
    t_pred = t_pred.ravel().reshape(feature1.shape)
    log.debug("t", t=t_pred)
    t_pred = (t_pred > 0.5).astype(int)
    display = DecisionBoundaryDisplay(xx0=feature1, xx1=feature2, response=t_pred)
    display.plot()

    display.ax_.scatter(data.x[:, 0], data.x[:, 1], c=data.t, edgecolor="black")

    plt.title("Spiral Classification - Decision Boundary ")

    settings.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = settings.output_dir / "hw02_plt.pdf"
    plt.savefig(output_path)
    log.info("Saved plot", path=str(output_path))
