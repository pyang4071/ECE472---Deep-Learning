from dataclasses import InitVar, dataclass, field

import numpy as np


@dataclass
class Data:
    rng: InitVar[np.random.Generator]
    num_samples: int
    sigma: float
    bias: float
    x: np.ndarray = field(init=False)
    y: np.ndarray = field(init=False)
    index: np.ndarray = field(init=False)

    def __post_init__(self, rng: np.random.Generator):
        """Generate synthetic data"""
        self.index = np.arange(self.num_samples)
        self.x = rng.uniform(0, 1, size=self.num_samples)
        clean_y = np.sin(2 * np.pi * self.x) + self.bias
        self.y = rng.normal(loc=clean_y, scale=self.sigma)

    def get_batch(self, rng: np.random.Generator, batch_size: int):
        """get random subset of data for training"""
        choices = rng.choice(self.index, size=batch_size)
        return self.x[choices], self.y[choices].flatten()
