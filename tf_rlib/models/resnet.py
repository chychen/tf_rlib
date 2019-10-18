import tensorflow as tf
from tf_rlib import layers, blocks, models
from absl import flags
from absl import logging

FLAGS = flags.FLAGS


class ResBlock(blocks.Block):
    outchannel_ratio = 1

    def __init__(self, filters, strides=1):
        super(ResBlock, self).__init__(filters, strides=strides)
        self.strides = strides
        self.bk1 = blocks.BasicBlock(filters,
                                     3,
                                     strides=strides,
                                     preact=False,
                                     use_norm=True,
                                     use_act=True)
        self.bk2 = blocks.BasicBlock(filters,
                                     3,
                                     strides=1,
                                     preact=False,
                                     use_norm=True,
                                     use_act=False)
        self.act = layers.Act()

        if self.strides != 1:
            self.shortcut = blocks.BasicBlock(filters,
                                              1,
                                              strides=strides,
                                              preact=False,
                                              use_norm=True,
                                              use_act=False)
        else:
            self.shortcut = None

    def call(self, x):
        out = self.bk1(x)
        out = self.bk2(out)
        if self.shortcut is None:
            out = out + x
        else:
            out = out + self.shortcut(x)
        out = self.act(out)
        return out


class ResNet_Cifar10(models.Model):
    def __init__(self, num_blocks):
        super(ResNet_Cifar10, self).__init__()
        self.out_dim = FLAGS.out_dim
        self.depth = FLAGS.depth
        self.block = ResBlock
        #         if FLAGS.bottleneck:
        #             self.block = ResBottleneck
        #         else:
        #             self.block = ResBlock

        self.in_filters = 64
        self.head = blocks.BasicBlock(self.in_filters,
                                      3,
                                      strides=1,
                                      preact=False,
                                      use_norm=True,
                                      use_act=False)

        self.all_groups = [
            self._build_group(self.in_filters, num_blocks[0], strides=1)
        ]
        for i in range(1, len(num_blocks)):
            self.all_groups.append(
                self._build_group(self.in_filters * (2**i),
                                  num_blocks[i],
                                  strides=2))

        self.gpool = layers.GlobalPooling()
        self.flatten = tf.keras.layers.Flatten()
        self.dense = layers.Dense(self.out_dim, activation=None, use_bias=True)

    def _build_group(self, filters, num_block, strides):
        all_blocks = []
        all_blocks.append(self.block(filters, strides=strides))
        for _ in range(1, num_block):
            all_blocks.append(self.block(filters, strides=1))
        return tf.keras.Sequential(all_blocks)

    def call(self, x):
        x = self.head(x)
        for group in self.all_groups:
            x = group(x)
        x = self.gpool(x)
        x = self.flatten(x)
        x = self.dense(x)
        return x
