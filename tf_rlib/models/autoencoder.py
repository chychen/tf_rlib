import tensorflow as tf
from tf_rlib import layers, blocks, models
from absl import flags, logging

FLAGS = flags.FLAGS
LOGGER = logging.get_absl_logger()

import tensorflow as tf
from tf_rlib import layers, blocks


class AE(models.Model):
    def __init__(self):
        super(AE, self).__init__()
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

        self.latent_dim = FLAGS.latent_dim
        self.depth = FLAGS.depth
        self.down_block = blocks.ResBlock
        self.up_block = blocks.UpBlock

        self.head = blocks.BasicBlock(self.in_filters,
                                      self.in_ks,
                                      strides=self.in_strides,
                                      preact=False,
                                      use_norm=True,
                                      use_act=self.in_act_pool)
        if self.in_act_pool:
            self.in_pool = tf.keras.layers.AveragePooling(
                pool_size=(3, ) * FLAGS.dim, strides=2, padding=FLAGS.padding)

        encoder_blocks = []
        filters = self.in_filters
        for _ in range(self.groups):
            filters = filters * 2
            encoder_blocks.append(blocks.ResBlock(filters, strides=2))
        self.encoder = self.sequential(encoder_blocks)
        self.latent_encoder = blocks.BasicBlock(self.latent_dim,
                                                3,
                                                strides=1,
                                                preact=True,
                                                use_norm=True,
                                                use_act=False)
        decoder_blocks = []
        for _ in range(self.groups):
            filters = filters // 2
            decoder_blocks.append(
                blocks.UpBlock(filters, with_concat=False, up_size=2))
        self.decoder = self.sequential(decoder_blocks)
        self.tail = blocks.BasicBlock(self.out_dim,
                                      3,
                                      strides=1,
                                      preact=True,
                                      use_norm=True,
                                      use_act=True)
        self.tanh = tf.keras.layers.Activation(tf.keras.activations.tanh)

    def call(self, x):
        x = self.head(x)
        if self.in_act_pool:
            x = self.in_pool(x)
        x = self.encoder(x)
        latent_code = self.latent_encoder(x)
        out = self.decoder(latent_code)
        out = self.tail(out)
        out = self.tanh(out)
        return latent_code, out
