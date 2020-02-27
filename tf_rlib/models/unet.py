import tensorflow as tf
from tf_rlib import layers, blocks, models
from absl import flags, logging

FLAGS = flags.FLAGS
LOGGER = logging.get_absl_logger()

import tensorflow as tf
from tf_rlib import layers, blocks


class UNet(models.Model):
    def __init__(self, init_filters=32, depth=4):
        super(UNet, self).__init__()
        self.out_dim = FLAGS.out_dim
        self.init_filters = init_filters
        self.depth = depth
        self.head = self._input_blocks()
        self.down_group = self._get_down_blocks()
        self.up_group = self._get_up_blocks()
        self.out = blocks.BasicBlock(self.out_dim,
                                     1,
                                     use_norm=False,
                                     use_bias=True,
                                     use_act=False)

    def _input_blocks(self):
        all_blocks = [
            blocks.BasicBlock(self.init_filters,
                              3,
                              strides=1,
                              preact=False,
                              use_norm=True,
                              use_act=True),
            blocks.ResBlock(self.init_filters)
        ]
        return self.sequential(all_blocks)

    def _get_down_blocks(self):
        group = []
        for i in range(self.depth):
            down_blocks = [
                blocks.BasicBlock(self.init_filters * 2**(i + 1),
                                  1,
                                  strides=1,
                                  use_norm=False,
                                  use_act=False),
                blocks.DownBlock(self.init_filters * 2**(i + 1))
            ]
            group.append(self.sequential(down_blocks))
        return group

    def _get_up_blocks(self):
        group = []
        for i in range(self.depth):
            up_blocks = [
                blocks.UpBlock(self.init_filters * 2**(self.depth - i - 1),
                               with_concat=True)
            ]
            group.append(self.sequential(up_blocks))
        return group

    def call(self, x):
        x = self.head(x)
        feat_maps = [x]
        for down in self.down_group:
            x = down(x)
            feat_maps.append(x)
        for up, map_ in zip(self.up_group, feat_maps[::-1][1:]):
            x = up([x, map_])
        x = self.out(x)
        x = tf.nn.softmax(x)
        return x
