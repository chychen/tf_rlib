import tensorflow as tf
from absl import flags
from absl import logging
import numpy as np

FLAGS = flags.FLAGS


class Model(tf.keras.Model):
    def __init__(self):
        super(Model, self).__init__()
        self.register = []

    @property
    def num_params(self):
        return sum(
            [np.prod(var.shape.as_list()) for var in self.trainable_variables])

    def sequential(self, blocks, register=True):
        if register:  # avoid gc clean up
            for bk in blocks:
                setattr(self, bk.name, bk)

        def seq_fn(x):
            for bk in blocks:
                x = getattr(self, bk.name)(x)
            return x

        return seq_fn
