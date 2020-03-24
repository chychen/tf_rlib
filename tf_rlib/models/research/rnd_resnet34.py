import tensorflow as tf
import tensorflow_hub as hub
from tf_rlib import layers, blocks, models
from absl import flags, logging

FLAGS = flags.FLAGS
LOGGER = logging.get_absl_logger()


class Predictor(models.Model):
    def __init__(self):
        super(Predictor, self).__init__()
        self.resnet34 = models.ResNet34(
            feature_mode=True, preact=False,
            last_norm=False)  #TODO remove the activation
        self.append_blocks = self._build_append_blocks()

    def _build_append_blocks(self):
        layer_list = [
            #             blocks.ResBlock(1024,
            #                            strides=1,
            #                            preact=False,
            #                            last_norm=False,
            #                            shortcut_type='project'),
            #             blocks.ResBlock(512,
            #                            strides=1,
            #                            preact=False,
            #                            last_norm=False,
            #                            shortcut_type='project'),
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
            tf.keras.layers.Flatten()
        ]
        return self.sequential(layer_list)

    def call(self, x):
        x = self.resnet34(x)
        x = self.append_blocks(x)
        return x


class RandomNet(models.Model):
    def __init__(self):
        super(RandomNet, self).__init__()
        self.resnet34 = models.ResNet34(
            feature_mode=True, preact=False,
            last_norm=False)  #TODO remove the activation
        self.append_blocks = self._build_append_blocks()

    def _build_append_blocks(self):
        layer_list = [
            #             blocks.ResBlock(1024,
            #                            strides=1,
            #                            preact=False,
            #                            last_norm=False,
            #                            shortcut_type='project'),
            #             blocks.ResBlock(512,
            #                            strides=1,
            #                            preact=False,
            #                            last_norm=False,
            #                            shortcut_type='project'),
            blocks.BasicBlock(32,
                              3,
                              strides=1,
                              preact=False,
                              use_norm=False,
                              use_act=False),
            tf.keras.layers.Flatten()
        ]
        return self.sequential(layer_list)

    def call(self, x):
        x = self.resnet34(x)
        x = self.append_blocks(x)
        return x


# class Predictor(models.Model):
#     def __init__(self):
#         super(Predictor, self).__init__()
#         self.append_blocks = self._build_append_blocks()

#     def _build_append_blocks(self):
#         layer_list = [
#             tf.keras.layers.Flatten(),
#             layers.Dense(20),
#             layers.Norm(),
#             layers.Act(),
#             layers.Dense(1),
#             layers.Act(),
#         ]
#         return self.sequential(layer_list)

#     def call(self, x):
#         x = self.append_blocks(x)
#         return x


# class RandomNet(models.Model):
#     def __init__(self):
#         super(RandomNet, self).__init__()
#         self.append_blocks = self._build_append_blocks()

#     def _build_append_blocks(self):
#         layer_list = [
#             tf.keras.layers.Flatten(),
#             layers.Dense(20),
#             layers.Norm(),
#             layers.Act(),
#             layers.Dense(1),
#             layers.Act(),
#         ]
#         return self.sequential(layer_list)

#     def call(self, x):
#         x = self.append_blocks(x)
#         return x
