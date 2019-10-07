import tensorflow as tf
from tensorflow.keras import layers
from absl import flags

FLAGS = flags.FLAGS


class ConvNd(tf.keras.layers.Layer):
    def __init__(self,
                 filters,
                 ks,
                 padding='same',
                 strides=1,
                 use_bias=False,
                 transpose=False):
        super(ConvNd, self).__init__()

        if transpose:
            conv_op = layers.__dict__['Conv{}DTranspose'.format(FLAGS.dim)]
            self.conv_op = conv_op(filters,
                                   ks,
                                   strides=strides,
                                   padding=padding,
                                   use_bias=use_bias)
        else:
            conv_op = layers.__dict__['Conv{}D'.format(FLAGS.dim)]
            self.conv_op = conv_op(filters,
                                   ks,
                                   strides=strides,
                                   padding=padding,
                                   use_bias=use_bias)

    def call(self, x):
        x = self.conv_op(x)
        return x
