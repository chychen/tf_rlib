import tensorflow as tf
from tf_rlib.layers import *


class BasicBlock(tf.keras.layers.Layer):
    def __init__(self,
                 filters,
                 ks,
                 padding='same',
                 strides=1,
                 preact=False,
                 use_norm=True,
                 use_act=True,
                 transpose=False):
        super(BasicBlock, self).__init__()

        self.preact = preact
        self.use_norm = use_norm
        self.use_act = use_act

        use_bias = False if self.preact else True
        if self.use_norm:
            self.act = Act()
        if self.use_act:
            self.bn = Norm()

        self.conv = ConvNd(filters,
                           ks,
                           strides=strides,
                           padding=padding,
                           use_bias=use_bias)

    def call(self, x):
        if self.preact:
            x = self.bn(x) if self.use_norm else x
            x = self.act(x)
            x = self.conv(x)
        else:
            x = self.conv(x)
            x = self.bn(x) if self.use_norm else x
            x = self.act(x)
