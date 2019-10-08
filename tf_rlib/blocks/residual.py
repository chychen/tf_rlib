import tensorflow as tf
from tf_rlib import layers, blocks
from absl import flags

FLAGS = flags.FLAGS


def shortcut_padding(out, x, downsample):
    if downsample is not None:
        shortcut = downsample(x)
    else:
        shortcut = x

    residual_channel = out.shape[-1]
    shortcut_channel = shortcut.shape[-1]

    if residual_channel != shortcut_channel:
        padding = [
            [0, 0],
        ] * (FLAGS.dim + 1) + [
            [0, residual_channel - shortcut_channel],
        ]
        out = out + tf.pad(shortcut, padding, "CONSTANT")
    else:
        out = out + shortcut
    return out


class ResBlock(blocks.Block):
    outchannel_ratio = 1

    def __init__(self, filters, strides=1):
        super(ResBlock, self).__init__(filters, strides=strides)
        self.strides = strides
        self.bk1 = blocks.BasicBlock(filters,
                                     3,
                                     strides=strides,
                                     preact=True,
                                     use_norm=True,
                                     use_act=False)
        self.bk2 = blocks.BasicBlock(filters,
                                     3,
                                     strides=1,
                                     preact=True,
                                     use_norm=True,
                                     use_act=True)
        self.bn = layers.Norm()
        self.downsample = layers.Pooling(
            pool_size=strides) if strides != 1 else None

    def call(self, x):
        out = self.bk1(x)
        out = self.bk2(out)
        out = self.bn(out)
        out = shortcut_padding(out, x, self.downsample)
        return out


class ResBottleneck(blocks.Block):
    outchannel_ratio = 4

    def __init__(self, filters, strides=1):
        super(ResBottleneck, self).__init__(filters, strides=strides)
        self.strides = strides
        self.bk1 = blocks.BasicBlock(filters,
                                     1,
                                     strides=1,
                                     preact=True,
                                     use_norm=True,
                                     use_act=False)
        self.bk2 = blocks.BasicBlock(filters,
                                     3,
                                     strides=strides,
                                     preact=True,
                                     use_norm=True,
                                     use_act=True)
        self.bk3 = blocks.BasicBlock(filters * ResBottleneck.outchannel_ratio,
                                     1,
                                     strides=1,
                                     preact=True,
                                     use_norm=True,
                                     use_act=True)
        self.bn = layers.Norm()
        self.downsample = layers.Pooling(
            pool_size=strides) if strides != 1 else None

    def call(self, x):
        out = self.bk1(x)
        out = self.bk2(out)
        out = self.bk3(out)
        out = self.bn(out)
        out = shortcut_padding(out, x, self.downsample)
        return out
