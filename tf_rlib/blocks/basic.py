import tensorflow as tf
from tf_rlib import layers


class BasicBlock(tf.keras.layers.Layer):
    def __init__(self,
                 filters,
                 ks,
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
            self.bn = layers.Norm()
        if self.use_act:
            self.act = layers.Act()
        self.conv = layers.Conv(filters,
                                ks,
                                strides=strides,
                                use_bias=use_bias)

    def call(self, x):
        if self.preact:
            if self.use_norm:
                x = self.bn(x)
            if self.use_act:
                x = self.act(x)
            x = self.conv(x)
        else:
            x = self.conv(x)
            if self.use_norm:
                x = self.bn(x)
            if self.use_act:
                x = self.act(x)
        return x
