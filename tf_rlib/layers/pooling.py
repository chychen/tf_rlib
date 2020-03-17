import tensorflow as tf
from tensorflow.keras import layers
from absl import flags

FLAGS = flags.FLAGS


class Pooling(tf.keras.layers.Layer):
    def __init__(self, pool_size=2):
        super(Pooling, self).__init__()
        self.pool_size = pool_size
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
        if FLAGS.padding == 'same_symmetric':
            paddings = [[0, 0]]
            for d in range(FLAGS.dim):
                res = x.shape[d + 1] % self.pool_size
                if res != 0:
                    paddings.append([0, res])
                else:
                    paddings.append([0, 0])
            paddings.append([0, 0])
            x = tf.pad(x, paddings, mode='SYMMETRIC')
        x = self.pooling_op(x)
        return x


class ShortcutPooling(tf.keras.layers.Layer):
    def __init__(self, pool_size=2):
        super(ShortcutPooling, self).__init__()
        self.pool_size = pool_size
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
        if FLAGS.padding == 'same_symmetric':
            paddings = [[0, 0]]
            for d in range(FLAGS.dim):
                res = x.shape[d + 1] % self.pool_size
                if res != 0:
                    paddings.append([0, res])
                else:
                    paddings.append([0, 0])
            paddings.append([0, 0])
            x = tf.pad(x, paddings, mode='SYMMETRIC')
        x = self.pooling_op(x)
        return x


class GlobalPooling(tf.keras.layers.Layer):
    def __init__(self, pool_size=2):
        super(GlobalPooling, self).__init__()
        self.pool_size = pool_size
        if FLAGS.global_pooling == 'GlobalAveragePooling' or FLAGS.global_pooling == 'GlobalMaxPooling':
            pooling_op = layers.__dict__[FLAGS.global_pooling +
                                         '{}D'.format(FLAGS.dim)]
            self.pooling_op = pooling_op()
        else:
            raise ValueError

    def call(self, x):
        x = self.pooling_op(x)
        return x


class ShortcutPoolingPadding(tf.keras.layers.Layer):
    def __init__(self, pool_size=2):
        super(ShortcutPoolingPadding, self).__init__()
        downsample = ShortcutPooling(pool_size=pool_size)
        self.pooling_op = lambda out, x: self.shortcut_padding(
            out, x, downsample)

    def call(self, out, x):
        return self.pooling_op(out, x)

    def shortcut_padding(self, out, x, downsample):
        shortcut = downsample(x)

        residual_channel = out.shape[-1]
        shortcut_channel = shortcut.shape[-1]

        if residual_channel != shortcut_channel:
            padding = [
                [0, 0],
            ] * (FLAGS.dim + 1) + [
                [0, tf.abs(residual_channel - shortcut_channel)],
            ]
            shortcut = tf.pad(shortcut, padding, "CONSTANT")
        return shortcut
