import tensorflow as tf
from tf_rlib import layers, blocks
from absl import flags

FLAGS = flags.FLAGS


class ResBlock(blocks.Block):
    outchannel_ratio = 1

    def __init__(self,
                 filters,
                 strides=1,
                 preact=True,
                 last_norm=True,
                 pool=False,
                 shortcut_type='project'):
        super(ResBlock, self).__init__(filters, strides=strides)
        self.strides = strides
        self.last_norm = last_norm
        self.preact = preact
        self.shortcut_type = shortcut_type
        self.pool = pool
        if self.pool:
            self.pool_lay = layers.Pooling(pool_size=strides)
        self.bk1 = blocks.BasicBlock(filters,
                                     3,
                                     strides=strides if not pool else 1,
                                     preact=preact,
                                     use_norm=True,
                                     use_act=False)
        self.bk2 = blocks.BasicBlock(filters,
                                     3,
                                     strides=1,
                                     preact=preact,
                                     use_norm=True,
                                     use_act=True,
                                     last_norm=last_norm)
        if self.shortcut_type == 'pad':
            if pool:
                raise ValueError
            self.shortcut = layers.ShortcutPoolingPadding(pool_size=strides)
        elif self.shortcut_type == 'project':
            self.shortcut = blocks.BasicBlock(
                filters,
                1,
                strides=strides if not pool else 1,
                preact=preact,
                use_norm=True,
                use_act=False,
                last_norm=last_norm)

    def call(self, x):
        out = self.pool_lay(x) if self.pool else x
        out = self.bk1(out)
        out = self.bk2(out)
        if self.shortcut_type == 'pad':
            shortcut = self.shortcut(out, x)
        elif self.shortcut_type == 'project':
            shortcut = self.shortcut(x)
        else:
            shortcut = x
        out = out + shortcut
        return out


class ResBottleneck(blocks.Block):
    outchannel_ratio = 4

    def __init__(self,
                 filters,
                 strides=1,
                 preact=True,
                 last_norm=True,
                 pool=False,
                 shortcut_type=None):
        super(ResBottleneck, self).__init__(filters, strides=strides)
        self.strides = strides
        self.last_norm = last_norm
        self.shortcut_type = shortcut_type
        self.pool = pool
        if self.pool:
            self.pool_lay = layers.Pooling(pool_size=strides)
        self.bk1 = blocks.BasicBlock(filters,
                                     1,
                                     strides=1,
                                     preact=preact,
                                     use_norm=True,
                                     use_act=False)
        self.bk2 = blocks.BasicBlock(filters,
                                     3,
                                     strides=strides if not pool else 1,
                                     preact=preact,
                                     use_norm=True,
                                     use_act=True)
        self.bk3 = blocks.BasicBlock(filters * ResBottleneck.outchannel_ratio,
                                     1,
                                     strides=1,
                                     preact=preact,
                                     use_norm=True,
                                     use_act=True,
                                     last_norm=last_norm)
        if self.shortcut_type == 'pad':
            if pool:
                raise ValueError
            self.shortcut_pad = layers.ShortcutPoolingPadding(
                pool_size=strides)
        elif self.shortcut_type == 'project':
            self.shortcut = blocks.BasicBlock(
                filters * ResBottleneck.outchannel_ratio,
                1,
                strides=strides if not pool else 1,
                preact=preact,
                use_norm=True,
                use_act=False,
                last_norm=last_norm)

    def call(self, x):
        out = self.pool_lay(x) if self.pool else x
        out = self.bk1(out)
        out = self.bk2(out)
        out = self.bk3(out)
        if self.shortcut_type == 'pad':
            shortcut = self.shortcut(out, x)
        elif self.shortcut_type == 'project':
            shortcut = self.shortcut(x)
        else:
            shortcut = x
        out = out + shortcut
        return out
