import tensorflow as tf
from absl import flags
from absl import logging
import numpy as np

FLAGS = flags.FLAGS


class Model(tf.keras.Model):
    def __init__(self, preprocessing_layer=None, **kwargs):
        super(Model, self).__init__()
        self.register = []
        self.preprocessing_layer = preprocessing_layer
        self._init()

    def call(self, x):
        if self.preprocessing_layer is not None:
            x = self.preprocessing_layer(x)
        x = self._call(x)
        return x

    def _init(self):
        raise NotImplementedError

    def _call(self, x):
        raise NotImplementedError

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
