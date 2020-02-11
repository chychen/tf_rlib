import tensorflow as tf
import tensorflow_hub as hub
from tf_rlib import layers, blocks, models
from absl import flags, logging

FLAGS = flags.FLAGS
LOGGER = logging.get_absl_logger()


class Encoder(models.Model):
    def __init__(self):
        """
        Arguments:
            input_shape: by default, (224, 224) is as same as the input shape of imagenet pretrained model
        """
        super(Encoder, self).__init__()
        self.encoder = self._build_encoder()

    def _build_encoder(self):
        layer_list = [
            layers.Conv(16, ks=4, strides=2),
            layers.Norm(),
            layers.Conv(64, ks=4, strides=2),
            layers.Norm(),
            layers.Conv(256, ks=4, strides=2),
            layers.Norm(),
            layers.Conv(512, ks=4, strides=2)
        ]
        return self.sequential(layer_list)

    def call(self, x):
        x = self.encoder(x)
        return x
