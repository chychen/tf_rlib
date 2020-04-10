import tensorflow as tf
from tf_rlib import layers, blocks
from absl import flags

FLAGS = flags.FLAGS


class BasicBlock(blocks.Block):
    def __init__(self,
                 filters,
                 ks,
                 strides=1,
                 preact=False,
                 use_norm=True,
                 use_act=True,
                 use_bias=False,
                 last_norm=False,
                 transpose=False):
        super(BasicBlock, self).__init__(filters, strides=strides)
        self.preact = preact
        self.use_norm = use_norm
        self.use_act = use_act
        self.last_norm = last_norm
        if self.use_norm:
            norm_scale = False if 'relu' in FLAGS.conv_act.lower() else True
            self.bn = layers.Norm(scale=norm_scale)
        if self.use_act:
            self.act = layers.Act()
        self.conv = layers.Conv(filters,
                                ks,
                                strides=strides,
                                use_bias=use_bias,
                                transpose=transpose)
        if self.last_norm:
            self.last_bn = layers.Norm()

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
        if self.last_norm:
            x = self.last_bn(x)
        return x
