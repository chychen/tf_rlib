import tensorflow as tf
import tensorflow_hub as hub
from tf_rlib import layers, blocks, models
from absl import flags, logging

FLAGS = flags.FLAGS
LOGGER = logging.get_absl_logger()

# ResNet34
class Predictor34(models.Model):
    def __init__(self):
        super(Predictor34, self).__init__()
        self.resnet = models.ResNet34(
            feature_mode=True, preact=False,
            last_norm=False)  # NOTE: activation exist in the end
        self.append_blocks = self._build_append_blocks()

    def _build_append_blocks(self):
        layer_list = [
            blocks.BasicBlock(64,
                              3,
                              strides=1,
                              preact=False,
                              use_norm=True,
                              use_act=False),
            blocks.BasicBlock(32,
                              3,
                              strides=1,
                              preact=False,
                              use_norm=True,
                              use_act=False),
            tf.keras.layers.Flatten(),
        ]
        return self.sequential(layer_list)

    def call(self, x):
        x = self.resnet(x)
        x = self.append_blocks(x)
        return x


class RandomNet34(models.Model):
    def __init__(self):
        super(RandomNet34, self).__init__()
        self.resnet = models.ResNet34(
            feature_mode=True, preact=False,
            last_norm=False)  # NOTE: activation exist in the end
        self.append_blocks = self._build_append_blocks()

    def _build_append_blocks(self):
        layer_list = [
            blocks.BasicBlock(32,
                              3,
                              strides=1,
                              preact=False,
                              use_norm=False,
                              use_act=False),
            tf.keras.layers.Flatten(),
        ]
        return self.sequential(layer_list)

    def call(self, x):
        x = self.resnet(x)
        x = self.append_blocks(x)
        return x

# ResNet18
class Predictor18(models.Model):
    def __init__(self):
        super(Predictor18, self).__init__()
        self.resnet = models.ResNet18(
            feature_mode=True, preact=False,
            last_norm=False)  # NOTE: activation exist in the end
        self.append_blocks = self._build_append_blocks()

    def _build_append_blocks(self):
        layer_list = [
            blocks.BasicBlock(64,
                              3,
                              strides=1,
                              preact=False,
                              use_norm=True,
                              use_act=False),
            blocks.BasicBlock(32,
                              3,
                              strides=1,
                              preact=False,
                              use_norm=True,
                              use_act=False),
            tf.keras.layers.Flatten(),
        ]
        return self.sequential(layer_list)

    def call(self, x):
        x = self.resnet(x)
        x = self.append_blocks(x)
        return x


class RandomNet18(models.Model):
    def __init__(self):
        super(RandomNet18, self).__init__()
        self.resnet = models.ResNet18(
            feature_mode=True, preact=False,
            last_norm=False)  # NOTE: activation exist in the end
        self.append_blocks = self._build_append_blocks()

    def _build_append_blocks(self):
        layer_list = [
            blocks.BasicBlock(32,
                              3,
                              strides=1,
                              preact=False,
                              use_norm=False,
                              use_act=False),
            tf.keras.layers.Flatten(),
        ]
        return self.sequential(layer_list)

    def call(self, x):
        x = self.resnet(x)
        x = self.append_blocks(x)
        return x
