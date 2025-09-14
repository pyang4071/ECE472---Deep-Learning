from dataclasses import InitVar, dataclass, field

import numpy as np

from .model import LinearModel


@dataclass
class Data:
    """Handles generation of synthetic data for linear regression."""

    model: LinearModel
    rng: InitVar[np.random.Generator]
    num_features: int
    num_samples: int
    sigma: float
    x: np.ndarray = field(init=False)
    y: np.ndarray = field(init=False)
    index: np.ndarray = field(init=False)

    def __post_init__(self, rng: np.random.Generator):
        """Generate synthetic data based on the model."""
        self.index = np.arange(self.num_samples)
        self.x = rng.uniform(0.1, 0.9, size=(self.num_samples, self.num_features))
        clean_y = self.x @ self.model.weights[:, np.newaxis] + self.model.bias
        self.y = rng.normal(loc=clean_y, scale=self.sigma)

    def get_batch(
        self, rng: np.random.Generator, batch_size: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Select random subset of examples for training batch."""
        choices = rng.choice(self.index, size=batch_size)

        return self.x[choices], self.y[choices].flatten()
