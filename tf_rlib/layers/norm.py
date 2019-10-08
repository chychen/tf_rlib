import tensorflow as tf
from tensorflow.keras import layers
from absl import flags

FLAGS = flags.FLAGS


class Norm(tf.keras.layers.Layer):
    def __init__(self):
        super(Norm, self).__init__()
        norm_op = layers.__dict__[FLAGS.conv_norm]
        if FLAGS.conv_norm == 'BatchNormalization':
            self.norm_op = norm_op(epsilon=FLAGS.bn_epsilon,
                                   momentum=FLAGS.bn_momentum)
        else:
            raise ValueError

    def call(self, x):
        x = self.norm_op(x)
        return x
