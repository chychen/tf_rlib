import tensorflow as tf
from tf_rlib import layers, blocks
from absl import flags

FLAGS = flags.FLAGS


def shortcut_padding(out, x, downsample):
    shortcut = downsample(x)

    residual_channel = out.shape[-1]
    shortcut_channel = shortcut.shape[-1]

    if residual_channel != shortcut_channel:
        padding = [
            [0, 0],
        ] * (FLAGS.dim + 1) + [
            [0, residual_channel - shortcut_channel],
        ]
        shortcut = tf.pad(shortcut, padding, "CONSTANT")
    return shortcut


class ResBlock(blocks.Block):
    outchannel_ratio = 1

    def __init__(self, filters, strides=1, preact=True, last_norm=True, shortcut_type='pad'):
        super(ResBlock, self).__init__(filters, strides=strides)
        self.strides = strides
        self.last_norm = last_norm
        self.shortcut_type= shortcut_type
        self.bk1 = blocks.BasicBlock(filters,
                                     3,
                                     strides=strides,
                                     preact=preact,
                                     use_norm=True,
                                     use_act=False)
        self.bk2 = blocks.BasicBlock(filters,
                                     3,
                                     strides=1,
                                     preact=preact,
                                     use_norm=True,
                                     use_act=True)
        if self.last_norm:
            self.bn = layers.Norm()
        if strides != 1:
            if self.shortcut_type == 'pad':
                self.downsample = layers.Pooling(pool_size=strides)
            elif self.shortcut_type == 'project':
                self.shortcut = blocks.BasicBlock(filters,
                                                  1,
                                                  strides=strides,
                                                  preact=preact,
                                                  use_norm=True,
                                                  use_act=True)
        else:
            if self.shortcut_type == 'pad':
                self.downsample = lambda in_: in_
            elif self.shortcut_type == 'project':
                self.shortcut = lambda in_: in_

                
    def call(self, x):
        out = self.bk1(x)
        out = self.bk2(out)
        if self.last_norm:
            out = self.bn(out)
        if self.shortcut_type == 'pad':
            out = out + shortcut_padding(out, x, self.downsample)
        elif self.shortcut_type == 'project':
            out = out + self.shortcut(x)
        return out


class ResBottleneck(blocks.Block):
    outchannel_ratio = 4

    def __init__(self, filters, strides=1, preact=True, last_norm=True):
        super(ResBottleneck, self).__init__(filters, strides=strides)
        self.strides = strides
        self.last_norm = last_norm
        self.bk1 = blocks.BasicBlock(filters,
                                     1,
                                     strides=1,
                                     preact=preact,
                                     use_norm=True,
                                     use_act=False)
        self.bk2 = blocks.BasicBlock(filters,
                                     3,
                                     strides=strides,
                                     preact=preact,
                                     use_norm=True,
                                     use_act=True)
        self.bk3 = blocks.BasicBlock(filters * ResBottleneck.outchannel_ratio,
                                     1,
                                     strides=1,
                                     preact=preact,
                                     use_norm=True,
                                     use_act=True)
        if self.last_norm:
            self.bn = layers.Norm()
        self.downsample = layers.Pooling(
            pool_size=strides) if strides != 1 else None

    def call(self, x):
        out = self.bk1(x)
        out = self.bk2(out)
        out = self.bk3(out)
        if self.last_norm:
            out = self.bn(out)
        out = shortcut_padding(out, x, self.downsample)
        return out
