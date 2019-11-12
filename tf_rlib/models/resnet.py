import tensorflow as tf
from tf_rlib import layers, blocks, models
from absl import flags
from absl import logging

FLAGS = flags.FLAGS
LOGGER = logging.get_absl_logger()


class ResNet_Cifar10(models.Model):
    def __init__(self, num_blocks):
        super(ResNet_Cifar10, self).__init__()
        self.out_dim = FLAGS.out_dim
        self.depth = FLAGS.depth
        self.block = blocks.ResBlock
        # TODO
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
        all_blocks.append(
            self.block(filters,
                       strides=strides,
                       preact=False,
                       last_norm=False,
                       shortcut_type='project'))
        LOGGER.debug('filters: {}'.format(filters))
        for _ in range(1, num_block):
            all_blocks.append(
                self.block(filters,
                           strides=1,
                           preact=False,
                           last_norm=False,
                           shortcut_type='project'))
            LOGGER.debug('filters: {}'.format(filters))
        return tf.keras.Sequential(all_blocks)

    def call(self, x):
        x = self.head(x)
        for group in self.all_groups:
            x = group(x)
        x = self.gpool(x)
        x = self.flatten(x)
        x = self.dense(x)
        return x
