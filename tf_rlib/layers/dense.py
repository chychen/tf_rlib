import tensorflow as tf
from tensorflow.keras import layers
from absl import flags

FLAGS = flags.FLAGS


class Dense(tf.keras.layers.Layer):
    def __init__(self, out_dim, activation=None, use_bias=True):
        super(Dense, self).__init__()
        self.dense_op = tf.keras.layers.Dense(
            out_dim,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=FLAGS.kernel_initializer,
            bias_initializer=FLAGS.bias_initializer)

    def call(self, x):
        return self.dense_op(x)
