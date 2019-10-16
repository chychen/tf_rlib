import tensorflow as tf
from tf_rlib import layers, blocks, models
from absl import flags
from absl import logging

FLAGS = flags.FLAGS


class PyramidNet(models.Model):
    def __init__(self):
        super(PyramidNet, self).__init__()
        self.out_dim = FLAGS.out_dim
        self.filters_mode = FLAGS.filters_mode
        if self.filters_mode == 'small':  # cifar
            self.in_filters = 16
            self.in_ks = 3
            self.in_strides = 1
            self.groups = 3
            self.in_act_pool = False
        elif self.filters_mode == 'large':  # imagenet
            self.in_filters = 64
            self.in_ks = 7
            self.in_strides = 2
            self.groups = 4
            self.in_act_pool = True
        else:
            raise ValueError
        self.depth = FLAGS.depth
        self.model_alpha = FLAGS.model_alpha
        if FLAGS.bottleneck:
            self.block = blocks.ResBottleneck
            group_blocks = int((self.depth - 2) / 3 / self.groups)
        else:
            self.block = blocks.ResBlock
            group_blocks = int((self.depth - 2) / 2 / self.groups)

        self.block_ratio = self.model_alpha / (self.groups * group_blocks)
        self.block_counter = 0

        self.head = blocks.BasicBlock(self.in_filters,
                                      self.in_ks,
                                      strides=self.in_strides,
                                      preact=False,
                                      use_norm=True,
                                      use_act=self.in_act_pool)
        if self.in_act_pool:
            self.in_pool = tf.keras.layers.AveragePooling(
                pool_size=(3, ) * FLAGS.dim, strides=2, padding=FLAGS.padding)

        self.all_groups = [self._build_pyramid_group(group_blocks, strides=1)]
        for _ in range(1, self.groups):
            self.all_groups.append(
                self._build_pyramid_group(group_blocks, strides=2))

        self.bn = layers.Norm()
        self.act = layers.Act()
        self.gpool = layers.GlobalPooling()
        self.flatten = tf.keras.layers.Flatten()
        self.dense = layers.Dense(self.out_dim, activation=None, use_bias=True)

    def _get_block_filters(self):
        self.block_counter = self.block_counter + 1
        filters = int(self.in_filters + self.block_counter * self.block_ratio)
        logging.info('block id: {}, filters: {}'.format(
            self.block_counter, filters))
        return filters

    def _build_pyramid_group(self, group_blocks, strides):
        all_blocks = []
        filters = self._get_block_filters()
        all_blocks.append(self.block(filters, strides=strides))
        for _ in range(1, group_blocks):
            filters = self._get_block_filters()
            all_blocks.append(self.block(filters, strides=1))
        return tf.keras.Sequential(all_blocks)

    def call(self, x):
        x = self.head(x)
        if self.in_act_pool:
            x = self.in_pool(x)
        for group in self.all_groups:
            x = group(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.gpool(x)
        x = self.flatten(x)
        x = self.dense(x)
        return x
