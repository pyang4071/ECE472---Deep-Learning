import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.inspection import DecisionBoundaryDisplay
import structlog
from tqdm import trange

from .config import PlottingSettings
from .data import Data_Spiral
from .model import NNXSpiralModel, SparseEncoder

log = structlog.get_logger()

font = {
    # "family": "Adobe Caslon Pro",
    "size": 10,
}

matplotlib.style.use("classic")
matplotlib.rc("font", **font)
cus_colors = plt.cm.coolwarm(np.linspace(0, 1, 2))


def plot_fit(
    model: NNXSpiralModel,
    data: Data_Spiral,
    settings: PlottingSettings,
):
    """Plots the linear fit and saves it to a file."""
    log.info("plotting")

    feature1, feature2 = np.meshgrid(
        np.linspace(
            data.x[:, 0].min() * 1.1,
            data.x[:, 0].max() * 1.1,
            settings.dpi * settings.figsize[1],
        ),
        np.linspace(
            data.x[:, 1].min() * 1.1,
            data.x[:, 1].max() * 1.1,
            settings.dpi * settings.figsize[0],
        ),
    )
    grid = np.vstack([feature1.ravel(), feature2.ravel()]).T
    t_pred = model(grid)
    t_pred = t_pred.ravel().reshape(feature1.shape)
    t_pred = (t_pred > 0.5).astype(int)
    display = DecisionBoundaryDisplay(xx0=feature1, xx1=feature2, response=t_pred)
    display.plot(cmap="coolwarm")

    display.ax_.scatter(data.x[:, 0], data.x[:, 1], c=data.t, edgecolor="black")

    plt.title("Spiral Classification - Decision Boundary ")
    plt.xlabel("x")
    ylab = plt.ylabel("y")
    ylab.set_rotation(0)

    settings.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = settings.output_dir / "hw07_plt.png"
    plt.savefig(output_path)
    log.info("Saved plot", path=str(output_path))


def plot_features(
    classifier: NNXSpiralModel,
    sparse_enc: SparseEncoder,
    settings: PlottingSettings,
    arg_max: bool = False,
):
    """plot a feature heat map"""
    feat = settings.enc_features

    x = np.linspace(-10, 10, settings.enc_dpi * settings.figsize[1])
    y = np.linspace(-10, 10, settings.enc_dpi * settings.figsize[0])

    X, Y = np.meshgrid(x, y)

    points = np.stack([X.ravel(), Y.ravel()], axis=1)

    Z_list = []

    batch_size = 8192
    for i in range(0, len(points), batch_size):
        neurons = classifier.get_features(points[i : i + batch_size])
        features = sparse_enc.get_features(neurons)
        if arg_max:
            feat_i = np.asarray(np.argmax(features, axis=1))
        else:
            feat_i = np.asarray(features[:, feat])

        Z_list.append(feat_i)

    Z = np.concatenate(Z_list, axis=0)
    log.info("Z shape", shape=Z.shape)
    h, w = X.shape

    if arg_max:
        Z = Z.reshape(h, w, 1)
        bar = trange(1)
    else:
        num_feat = len(feat)
        Z = Z.reshape(h, w, num_feat)
        # same shape as X and Y for plotting but multiple features

        for i in range(num_feat):
            if np.max(Z[:, :, i]) != 0:
                Z[:, :, i] = Z[:, :, i] / np.max(Z[:, :, i])  # normalize

        bar = trange(num_feat)

    for i in bar:
        plt.figure(figsize=settings.figsize)
        plt.imshow(Z[:, :, i], cmap="coolwarm", extent=(-10, 10, -10, 10))
        plt.colorbar(label="Feature activation level")
        plt.title(f"Feature {feat[i]} Heapmap")
        plt.xlabel("x")
        ylab = plt.ylabel("y")
        ylab.set_rotation(0)

        settings.output_dir.mkdir(parents=True, exist_ok=True)
        output = f"hw07/feature_{feat[i]}.png"
        output_path = settings.output_dir / output
        plt.savefig(output_path)
        # log.info("Saved plot", path=str(output_path))

        plt.close()
