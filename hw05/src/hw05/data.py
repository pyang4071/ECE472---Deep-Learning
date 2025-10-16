from dataclasses import InitVar, dataclass, field
import numpy as np
import structlog

import tensorflow as tf
import tensorflow_datasets as tfds

log = structlog.get_logger()


@dataclass
class Data:
    rng: InitVar[np.random.Generator]
    num_splits: InitVar[int]
    token_length: InitVar[int]
    max_token: InitVar[int]

    train_token: np.ndarray = field(init=False)
    train_label: np.ndarray = field(init=False)

    test_token: np.ndarray = field(init=False)
    test_label: np.ndarray = field(init=False)

    def __post_init__(
        self,
        rng,
        num_splits: int,
        token_length: int = 20,
        max_token: int = 1000,
    ):
        self.token_length = token_length
        self.max_token = max_token

        (x_train, x_test), ds_info = tfds.load(
            "ag_news_subset",
            split=["train", "test"],
            shuffle_files=True,
            as_supervised=True,
            with_info=True,
            download=True,
        )
        log.info("download data")

        train_texts = []
        self.train_label = []
        for text, label in tfds.as_numpy(x_train):
            train_texts.append(text.decode("utf-8"))
            self.train_label.append(label)
        self.train_label = np.array(self.train_label, dtype=int)

        test_texts = []
        self.test_label = []
        for text, label in tfds.as_numpy(x_test):
            test_texts.append(text.decode("utf-8"))
            self.test_label.append(label)
        self.test_label = np.array(self.test_label, dtype=int)

        log.info("data labels split")

        # vectorize all at once

        vectorization_layer = tf.keras.layers.TextVectorization(
            max_tokens=self.max_token,
            standardize="lower_and_strip_punctuation",
            split="whitespace",
            output_mode="int",
            output_sequence_length=self.token_length,
            encoding="utf-8",
            ngrams=2,
        )

        vectorization_layer.adapt(train_texts, batch_size=6000)

        # convert to integer
        self.train_token = vectorization_layer(train_texts).numpy()
        # shape (num_data, max_len)

        self.test_token = vectorization_layer(test_texts).numpy()

        log.info("vectorized")

        self.num_splits = num_splits
        self.indices = np.arange(len(self.train_label))
        rng.shuffle(self.indices)
        self.indices = np.array_split(self.indices, self.num_splits)
        # split into separate section

    def get_batch(self, rng: np.random.Generator, batch_size: int, val_set: int):
        if val_set >= self.num_splits:
            log.info("invalid val_set", val_set=val_set)

        index = np.concatenate(
            [self.indices[i] for i in range(self.num_splits) if i != val_set]
        ).astype(int)

        choices = rng.choice(index, size=batch_size).astype(int)
        log.debug("choices", choices=choices)

        return self.train_token[choices], self.train_label[choices]

    def get_validation(self, val_set: int):
        return self.train_token[self.indices[val_set]], self.train_label[
            self.indices[val_set]
        ]

    def get_test(self):
        return self.test_token, self.test_label
