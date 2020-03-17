import tensorflow as tf
from tf_rlib import layers, blocks, models
from absl import flags
from absl import logging

FLAGS = flags.FLAGS
LOGGER = logging.get_absl_logger()


class ResNet(models.Model):
    def __init__(self, num_blocks, feature_mode=False, preact=False, last_norm=False):
        """ By default, preact=False, last_norm=False means vanilla resnet.
        """
        super(ResNet, self).__init__()
        self.out_dim = FLAGS.out_dim
        self.depth = FLAGS.depth
        self.preact = preact
        self.last_norm = last_norm
        self.feature_mode = feature_mode
        if FLAGS.bottleneck:
            self.block = blocks.ResBottleneck
        else:
            self.block = blocks.ResBlock

        self.filters_mode = FLAGS.filters_mode
        if self.filters_mode == 'small':  # cifar
            self.in_ks = 3
            self.in_strides = 1
            self.in_act_pool = False
        elif self.filters_mode == 'large':  # imagenet
            self.in_ks = 7
            self.in_strides = 2
            self.in_act_pool = True
        else:
            raise ValueError

        self.in_filters = 64
        self.head = blocks.BasicBlock(self.in_filters,
                                      self.in_ks,
                                      strides=self.in_strides,
                                      preact=self.preact,
                                      use_norm=True,
                                      use_act=self.in_act_pool)

        if self.in_act_pool:
            self.in_pool = tf.keras.layers.AveragePooling(
                pool_size=(3, ) * FLAGS.dim, strides=2, padding=FLAGS.padding)

        self.all_groups = [
            self._build_group(self.in_filters, num_blocks[0], strides=1)
        ]
        for i in range(1, len(num_blocks)):
            self.all_groups.append(
                self._build_group(self.in_filters * (2**i),
                                  num_blocks[i],
                                  strides=2))

        if not self.feature_mode:
            self.gpool = layers.GlobalPooling()
            self.flatten = tf.keras.layers.Flatten()
            self.dense = layers.Dense(self.out_dim, activation=None, use_bias=True)

    def _build_group(self, filters, num_block, strides):
        all_blocks = []
        all_blocks.append(
            self.block(filters,
                       strides=strides,
                       preact=self.preact,
                       last_norm=self.last_norm,
                       shortcut_type='project'))
        LOGGER.debug('filters: {}'.format(filters))
        for _ in range(1, num_block):
            all_blocks.append(
                self.block(filters,
                           strides=1,
                           preact=self.preact,
                           last_norm=self.last_norm,
                           shortcut_type=None))
            LOGGER.debug('filters: {}'.format(filters))
        return tf.keras.Sequential(all_blocks)

    def call(self, x):
        x = self.head(x)
        if self.in_act_pool:
            x = self.in_pool(x)
        for group in self.all_groups:
            x = group(x)
        if not self.feature_mode:
            x = self.gpool(x)
            x = self.flatten(x)
            x = self.dense(x)
        return x


def ResNet18(feature_mode=False, preact=False, last_norm=False):
    """
    Total params: 11,187,914
    Trainable params: 11,178,186
    Non-trainable params: 9,728
    """
    FLAGS.depth = 18
    FLAGS.bottleneck = False
    m = ResNet([2, 2, 2, 2], feature_mode=feature_mode, preact=preact, last_norm=last_norm)
    return m


def ResNet34(feature_mode=False, preact=False, last_norm=False):
    """
    Total params: 21,303,498
    Trainable params: 21,286,346
    Non-trainable params: 17,152
    """
    FLAGS.depth = 34
    FLAGS.bottleneck = False
    m = ResNet([3, 4, 6, 3], feature_mode=feature_mode, preact=preact, last_norm=last_norm)
    return m


def ResNet50(feature_mode=False, preact=False, last_norm=False):
    """
    Total params: 23,573,962
    Trainable params: 23,520,842
    Non-trainable params: 53,120
    """
    FLAGS.depth = 50
    FLAGS.bottleneck = True
    m = ResNet([3, 4, 6, 3], feature_mode=feature_mode, preact=preact, last_norm=last_norm)
    return m
