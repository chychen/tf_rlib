import tensorflow as tf
from tensorflow.keras import layers
from absl import flags

FLAGS = flags.FLAGS


class UpSampling(tf.keras.layers.Layer):
    def __init__(self, up_size=2):
        super(UpSampling, self).__init__()
        upsampling_op = layers.__dict__['UpSampling' +
                                        '{}D'.format(FLAGS.dim)]
        self.upsampling_op = upsampling_op(size=up_size)

    def call(self, x):
        x = self.upsampling_op(x)
        return x


