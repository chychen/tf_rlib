import tensorflow as tf
from absl import flags
from absl import logging
import numpy as np

FLAGS = flags.FLAGS


class Model(tf.keras.Model):
    def __init__(self):
        super(Model, self).__init__()

    @property
    def num_params(self):
        return sum(
            [np.prod(var.shape.as_list()) for var in self.trainable_variables])
