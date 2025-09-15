from dataclasses import InitVar, dataclass, field
import numpy as np
import structlog


log = structlog.get_logger()


@dataclass
class Data_Spiral:
    rng: InitVar[np.random.Generator]
    num_features: int
    num_samples: int  # assume this value is always even
    sigma: float
    x: np.ndarray = field(init=False)
    t: np.ndarray = field(init=False)
    index: np.ndarray = field(init=False)

    def __post_init__(self, rng: np.random.Generator):
        """generate the spiral"""
        self.index = np.arange(self.num_samples)
        # generate data using polar coordinates
        # archimedes spiral
        # red spiral => r = 0.5(theta - pi/4)
        # blue spiral => r = 0.5(-theta + pi/4)
        # theta in (-7pi/4, 0)

        n = self.num_samples // 2

        # generate red
        theta1 = rng.uniform(-7 * np.pi / 2, 0, size=n)
        r1 = 0.5 * (2 * theta1 - np.pi / 2)
        x1 = r1 * np.cos(theta1)
        y1 = r1 * np.sin(theta1)
        log.debug("x1, y1", x1=x1, y1=y1)

        # generate blue
        theta2 = rng.uniform(-7 * np.pi / 2, 0, size=n)
        r2 = 0.5 * (-2 * theta2 + np.pi / 2)
        x2 = r2 * np.cos(theta2)
        y2 = r2 * np.sin(theta2)

        # combine
        self.x = np.vstack([np.column_stack([x1, y1]), np.column_stack([x2, y2])])

        self.x += rng.normal(loc=0, scale=self.sigma, size=self.x.shape)
        self.t = np.concatenate([np.ones(n), np.zeros(n)])

        log.debug("coodinates for data", data=self.x)

    def get_batch(
        self, rng: np.random.Generator, batch_size: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """get random subset"""
        choices = rng.choice(self.index, size=batch_size)
        return self.x[choices], self.t[choices]  # self.x is (batch_size, 2)
