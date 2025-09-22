from dataclasses import InitVar, dataclass, field
import numpy as np
import structlog

import tensorflow as tf


log = structlog.get_logger()


@dataclass
class Data_MNIST:
    rng: InitVar[np.random.Generator]
    split_ratio: float = 0.8
    x_train: np.ndarray = field(init=False)
    y_train: np.ndarray = field(init=False)
    x_val: np.ndarray = field(init=False)
    y_val: np.ndarray = field(init=False)
    x_test: np.ndarray = field(init=False)
    y_test: np.ndarray = field(init=False)
    index: np.ndarray = field(init=False)

    def __post_init__(self, rng: np.random.Generator):
        """get the MINST dataset"""
        (x_train_temp, y_train_temp), (self.x_test, self.y_test) = (
            tf.keras.datasets.mnist.load_data()
        )
        # normalize
        x_train_temp = x_train_temp / 255.0  # (num_samples, 28, 28)
        self.x_test = self.x_test / 255.0

        # add depth
        x_train_temp = np.expand_dims(x_train_temp, -1)  # (size, 28, 28, 1)
        self.x_test = np.expand_dims(self.x_test, -1)

        # split into train and validating
        indices = np.arange(len(y_train_temp))
        rng.shuffle(indices)
        split = int(self.split_ratio * len(y_train_temp))
        train_index = indices[:split]
        valid_index = indices[split:]

        self.x_train = x_train_temp[train_index]  # (amount, pic_height, pic_width, 1)
        self.y_train = y_train_temp[train_index]  # (amount,)

        self.x_val = x_train_temp[valid_index]
        self.y_val = y_train_temp[valid_index]

        self.index = np.arange(len(self.y_train))

    def get_batch(
        self, rng: np.random.Generator, batch_size: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """get random subset"""
        choices = rng.choice(self.index, size=batch_size)
        return self.x_train[choices], self.y_train[
            choices
        ]  # x_train is (batch size, 28, 28, 1)

    def get_validation(self):
        return self.x_val, self.y_val

    def get_test(self):
        return self.x_test, self.y_test
