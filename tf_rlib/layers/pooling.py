import tensorflow as tf
from tensorflow.keras import layers
from absl import flags

FLAGS = flags.FLAGS


class Pooling(tf.keras.layers.Layer):
    def __init__(self, pool_size=2):
        super(Pooling, self).__init__()
        if FLAGS.conv_pooling == 'AveragePooling' or FLAGS.conv_pooling == 'MaxPooling':
            pooling_op = layers.__dict__[FLAGS.conv_pooling +
                                         '{}D'.format(FLAGS.dim)]
            padding = 'valid' if FLAGS.padding == 'same_symmetric' else FLAGS.padding
            self.pooling_op = pooling_op(
                pool_size=pool_size,
                strides=None,  # == pool_size
                padding=padding)
        else:
            raise ValueError

    def call(self, x):
        x = self.pooling_op(x)
        return x


class ShortcutPooling(tf.keras.layers.Layer):
    def __init__(self, pool_size=2):
        super(ShortcutPooling, self).__init__()
        if FLAGS.shortcut_pooling == 'AveragePooling' or FLAGS.shortcut_pooling == 'MaxPooling':
            pooling_op = layers.__dict__[FLAGS.conv_pooling +
                                         '{}D'.format(FLAGS.dim)]
            padding = 'valid' if FLAGS.padding == 'same_symmetric' else FLAGS.padding
            self.pooling_op = pooling_op(
                pool_size=pool_size,
                strides=None,  # == pool_size
                padding=padding)
        else:
            raise ValueError

    def call(self, x):
        x = self.pooling_op(x)
        return x


class GlobalPooling(tf.keras.layers.Layer):
    def __init__(self, pool_size=2):
        super(GlobalPooling, self).__init__()
        if FLAGS.global_pooling == 'GlobalAveragePooling' or FLAGS.global_pooling == 'GlobalMaxPooling':
            pooling_op = layers.__dict__[FLAGS.global_pooling +
                                         '{}D'.format(FLAGS.dim)]
            self.pooling_op = pooling_op()
        else:
            raise ValueError

    def call(self, x):
        x = self.pooling_op(x)
        return x
