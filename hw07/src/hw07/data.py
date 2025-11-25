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
        # red spiral => r = theta
        # blue spiral => r = -theta
        # theta in (-7pi/2, 0)

        n = self.num_samples // 2

        # generate red
        theta1 = rng.uniform(-7 * np.pi / 2, 0, size=n)
        r1 = theta1
        x1 = r1 * np.cos(theta1)
        y1 = r1 * np.sin(theta1)
        log.debug("x1, y1", x1=x1, y1=y1)

        # generate blue
        theta2 = rng.uniform(-7 * np.pi / 2, 0, size=n)
        r2 = -theta2
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


class Data_Activations:
    def __init__(self, rng: np.random.Generator, model, num, act_dim):
        self.rng = rng

        points = rng.uniform(low=[-15, -15], high=[15, 15], size=(num, 2))

        data_act = np.zeros([0, act_dim])
        for i in range(0, num, 512):
            x = points[i : i + 512]
            x = model.get_features(x)
            data_act = np.vstack([data_act, x])

        log.debug("shape of act data", act=data_act.shape)

        # shuffle rows
        rng.shuffle(data_act)

        # each data is scaled
        # A = E[norm(x)**2]
        # l2norm**2 = sum of square of each term. Then we average over all sample
        # A = (1/num_samples) * sum of all terms squared
        A = (1 / num) * np.sum(data_act**2)
        # want some constant c where the x corresponding to cx is sqrt(num_features)
        # so c^2 * A = sqrt(num_features)
        c = np.sqrt(np.sqrt(act_dim) / A)
        self.data_act = data_act * c

        self.index = np.arange(num)

    def get_batch(
        self, rng: np.random.Generator, batch_size: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """get random subset"""

        choices = rng.choice(self.index, size=batch_size)
        return self.data_act[choices]
