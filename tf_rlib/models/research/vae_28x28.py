import tensorflow as tf
import tensorflow_hub as hub
from tf_rlib import layers, blocks, models
from absl import flags, logging

FLAGS = flags.FLAGS
LOGGER = logging.get_absl_logger()


class VAEEncoder(models.Model):
    def __init__(self):
        super(VAEEncoder, self).__init__()
        self.latent_dim = FLAGS.latent_dim
        self.inference_net = self._build_inference()

    def _build_inference(self):
        layer_list = [
            layers.Conv(32, ks=3, strides=2),
            layers.Act('relu'),
            layers.Conv(64, ks=3, strides=2),
            layers.Act('relu'),
            tf.keras.layers.Flatten(),
            layers.Dense(self.latent_dim + self.latent_dim)
        ]
        return self.sequential(layer_list)

    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=[tf.shape(mean)[0], FLAGS.latent_dim])
        return eps * tf.exp(logvar * .5) + mean

    def call(self, x):
        logits = self.inference_net(x)
        mean, logvar = tf.split(logits, num_or_size_splits=2, axis=1)
        z = self.reparameterize(mean, logvar)
        return mean, logvar, z


class VAEDecoder(models.Model):
    def __init__(self):
        super(VAEDecoder, self).__init__()
        self.latent_dim = FLAGS.latent_dim
        self.generative_net = self._build_generator()

    def _build_generator(self):
        layer_list = [
            layers.Dense(7 * 7 * 32),
            layers.Act('relu'),
            tf.keras.layers.Reshape([7, 7, 32]),
            layers.Conv(64, ks=3, strides=2, transpose=True),
            layers.Act('relu'),
            layers.Conv(32, ks=3, strides=2, transpose=True),
            layers.Act('relu'),
            layers.Conv(FLAGS.out_dim, ks=3, strides=1, transpose=True)
        ]
        return self.sequential(layer_list)

    def call(self, x):
        out = self.generative_net(x)
        return out
