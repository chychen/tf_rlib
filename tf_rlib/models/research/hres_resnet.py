""" Modified from ResNet
"""
import tensorflow as tf
from tf_rlib import layers, blocks, models
from absl import flags
from absl import logging

FLAGS = flags.FLAGS
LOGGER = logging.get_absl_logger()


class HResResNetTail(models.Model):
    def __init__(self, preact=False, last_norm=False):
        """ By default, preact=False, last_norm=False means vanilla resnet.
        """
        super(HResResNetTail, self).__init__()
        self.out_dim = FLAGS.out_dim
        self.preact = preact
        self.last_norm = last_norm

        self.tail = blocks.BasicBlock(FLAGS.out_dim,
                                      1,
                                      strides=1,
                                      preact=self.preact,
                                      use_norm=self.preact,
                                      use_act=self.preact,
                                      use_bias=True)

    def call(self, x):
        return self.tail(x)


class HResResNet(models.Model):
    def __init__(self,
                 num_blocks,
                 feature_mode=False,
                 preact=False,
                 last_norm=False):
        """ By default, preact=False, last_norm=False means vanilla resnet.
        """
        super(HResResNet, self).__init__()
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
        elif self.filters_mode == 'large':  # imagenet
            self.in_ks = 7
        else:
            raise ValueError

        self.in_filters = FLAGS.in_filters if FLAGS.in_filters else 64
        self.head = blocks.BasicBlock(self.in_filters,
                                      self.in_ks,
                                      strides=1,
                                      preact=False,
                                      use_norm=True,
                                      use_act=True)

        self.all_groups = [
            self._build_group(self.in_filters, num_blocks[0], strides=1)
        ]
        for i in range(1, len(num_blocks)):
            self.all_groups.append(
                self._build_group(self.in_filters * (2**i),
                                  num_blocks[i],
                                  strides=1))
        if not self.feature_mode:
            self.tail = HResResNetTail(self.preact, self.last_norm)

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
        for group in self.all_groups:
            x = group(x)
        if not self.feature_mode:
            x = self.tail(x)
        return x


def HResResNet18(feature_mode=False, preact=False, last_norm=False):
    """
    Total params: 11,187,914
    Trainable params: 11,178,186
    Non-trainable params: 9,728
    """
    FLAGS.depth = 18
    FLAGS.bottleneck = False
    m = HResResNet([2, 2, 2, 2],
                   feature_mode=feature_mode,
                   preact=preact,
                   last_norm=last_norm)
    return m


def HResResNet34(feature_mode=False, preact=False, last_norm=False):
    """
    Total params: 21,303,498
    Trainable params: 21,286,346
    Non-trainable params: 17,152
    """
    FLAGS.depth = 34
    FLAGS.bottleneck = False
    m = HResResNet([3, 4, 6, 3],
                   feature_mode=feature_mode,
                   preact=preact,
                   last_norm=last_norm)
    return m


def HResResNet50(feature_mode=False, preact=False, last_norm=False):
    """
    Total params: 23,573,962
    Trainable params: 23,520,842
    Non-trainable params: 53,120
    """
    FLAGS.depth = 50
    FLAGS.bottleneck = True
    m = HResResNet([3, 4, 6, 3],
                   feature_mode=feature_mode,
                   preact=preact,
                   last_norm=last_norm)
    return m


def HResWRN16_8(feature_mode=False, preact=False, last_norm=False):
    """ 
    """
    FLAGS.depth = 18
    FLAGS.bottleneck = False
    FLAGS.in_filters = 16 * 8
    m = HResResNet([2, 2, 2, 2],
                   feature_mode=feature_mode,
                   preact=preact,
                   last_norm=last_norm)
    return m
