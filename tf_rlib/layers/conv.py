import tensorflow as tf
from tensorflow.keras import layers
from absl import flags

FLAGS = flags.FLAGS


class Conv(tf.keras.layers.Layer):
    def __init__(self,
                 filters,
                 ks,
                 strides=1,
                 use_bias=False,
                 transpose=False):
        super(Conv, self).__init__()

        if transpose:
            conv_op = layers.__dict__['Conv{}DTranspose'.format(FLAGS.dim)]
        else:
            conv_op = layers.__dict__['Conv{}D'.format(FLAGS.dim)]

        self.ks = ks
        padding = 'valid' if FLAGS.padding == 'same_symmetric' else FLAGS.padding
        self.conv_op = conv_op(
            filters,
            ks,
            strides=strides,
            padding=padding,
            use_bias=use_bias,
            kernel_initializer=FLAGS.kernel_initializer,
            bias_initializer=FLAGS.bias_initializer,
            kernel_regularizer=tf.keras.regularizers.l1_l2(l1=FLAGS.l1,
                                                           l2=FLAGS.l2),
            bias_regularizer=tf.keras.regularizers.l1_l2(l1=FLAGS.l1,
                                                         l2=FLAGS.l2))

    def call(self, x):
        if FLAGS.padding == 'same_symmetric':
            paddings = [[0, 0]]
            for _ in range(FLAGS.dim):
                paddings.append([self.ks // 2, self.ks // 2])
            paddings.append([0, 0])
            x = tf.pad(x, paddings, mode='SYMMETRIC')
        x = self.conv_op(x)
        return x
