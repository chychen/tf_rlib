import tensorflow as tf
from tensorflow.keras import layers
from absl import flags

FLAGS = flags.FLAGS


class Conv(tf.keras.layers.Layer):
    def __init__(self, filters, ks, strides=1, use_bias=False,
                 transpose=False):
        super(Conv, self).__init__()

        if transpose:
            conv_op = layers.__dict__['Conv{}DTranspose'.format(FLAGS.dim)]
            self.conv_op = conv_op(
                filters,
                ks,
                strides=strides,
                padding=FLAGS.padding,
                use_bias=use_bias,
                kernel_initializer=FLAGS.kernel_initializer,
                bias_initializer=FLAGS.bias_initializer,
                kernel_regularizer=tf.keras.regularizers.l1_l2(l1=FLAGS.l1,
                                                               l2=FLAGS.l2),
                bias_regularizer=tf.keras.regularizers.l1_l2(l1=FLAGS.l1,
                                                             l2=FLAGS.l2))
        else:
            conv_op = layers.__dict__['Conv{}D'.format(FLAGS.dim)]
            self.conv_op = conv_op(
                filters,
                ks,
                strides=strides,
                padding=FLAGS.padding,
                use_bias=use_bias,
                kernel_initializer=FLAGS.kernel_initializer,
                bias_initializer=FLAGS.bias_initializer,
                kernel_regularizer=tf.keras.regularizers.l1_l2(l1=FLAGS.l1,
                                                               l2=FLAGS.l2),
                bias_regularizer=tf.keras.regularizers.l1_l2(l1=FLAGS.l1,
                                                             l2=FLAGS.l2))

    def call(self, x):
        x = self.conv_op(x)
        return x
