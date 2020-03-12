import tensorflow as tf
from tensorflow.keras import layers
import tensorflow_addons as tfa
from absl import flags

FLAGS = flags.FLAGS


class Norm(tf.keras.layers.Layer):
    def __init__(self, norm_type=None):
        """ By default with norm_type=None, the API is controled by FLAGS.conv_norm.
        """
        super(Norm, self).__init__()

        if norm_type is None:
            norm_type = FLAGS.conv_norm

        if norm_type in layers.__dict__:
            norm_op = layers.__dict__[norm_type]
        else:
            norm_op = tfa.layers.__dict__[norm_type]

        if norm_type == 'BatchNormalization':
            self.norm_op = norm_op(epsilon=FLAGS.bn_epsilon,
                                   momentum=FLAGS.bn_momentum)
        elif norm_type == 'InstanceNormalization' or norm_type == 'LayerNormalization':
            self.norm_op = norm_op()
        elif norm_type == 'GroupNormalization':
            self.norm_op = norm_op(groups=FLAGS.groups)
        else:
            raise ValueError

    def call(self, x):
        x = self.norm_op(x)
        return x
