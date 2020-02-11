import tensorflow as tf
import tensorflow_hub as hub
from tf_rlib import layers, blocks, models
from absl import flags, logging

FLAGS = flags.FLAGS
LOGGER = logging.get_absl_logger()

# class AEClassifier(models.Model):
#     """ AutoEncoder regularized CNN-based Classifier
#     Reference: [Active Authentication using an Autoencoder regularized CNN-based One-Class Classifier](https://arxiv.org/abs/1903.01031)
#     """
#     def __init__(self):
#         """
#         Arguments:
#             input_shape: by default, (224, 224) is as same as the input shape of imagenet pretrained model
#         """
#         super(AEClassifier, self).__init__()
#         self.first_norm = layers.Norm()
#         self.classifier = self._build_classifier()
#         self.decoder = self._build_decoder()

#     def _build_decoder(self):
#         layer_list = [
#             layers.Conv(256, ks=4, strides=2, transpose=True),
#             layers.Norm(),
#             layers.Conv(64, ks=4, strides=2, transpose=True),
#             layers.Norm(),
#             layers.Conv(16, ks=4, strides=2, transpose=True),
#             layers.Norm(),
#             layers.Conv(3, ks=4, strides=2, transpose=True),
#             layers.Act('Tanh')
#         ]
#         return self.sequential(layer_list)

#     def _build_classifier(self):
#         layer_list = [
#             layers.Conv(256, ks=3, strides=2),
#             layers.Norm(),
#             layers.GlobalPooling(),
#             tf.keras.layers.Flatten(),
#             layers.Dense(256),
#             layers.Norm(),
#             layers.Dense(1),
#             layers.Act('Sigmoid')
#         ]
#         return self.sequential(layer_list)

#     def call(self, x):
#         x = self.first_norm(x)
#         probs = self.classifier(x)
#         reconstruct = self.decoder(x)
#         return probs, reconstruct


class AEClassifier(models.Model):
    """ AutoEncoder regularized CNN-based Classifier
    Reference: [Active Authentication using an Autoencoder regularized CNN-based One-Class Classifier](https://arxiv.org/abs/1903.01031)
    """
    def __init__(self):
        """
        Arguments:
            input_shape: by default, (224, 224) is as same as the input shape of imagenet pretrained model
        """
        super(AEClassifier, self).__init__()
        self.first_norm = layers.Norm()
        #         self.dense = layers.Dense(14*14*8)
        self.reshape = tf.keras.layers.Reshape((8, 8, 32))
        self.classifier = self._build_classifier()
        self.decoder = self._build_decoder()

    def _build_decoder(self):
        layer_list = [
            blocks.BasicBlock(256,
                              ks=4,
                              strides=2,
                              transpose=True,
                              use_act=True),
            blocks.BasicBlock(64,
                              ks=4,
                              strides=2,
                              transpose=True,
                              use_act=True),
            blocks.BasicBlock(16,
                              ks=4,
                              strides=2,
                              transpose=True,
                              use_act=True),
            blocks.BasicBlock(3,
                              ks=4,
                              strides=2,
                              transpose=True,
                              use_act=False),
            layers.Act('Tanh')
        ]
        return self.sequential(layer_list)

    def _build_classifier(self):
        layer_list = [
            layers.Dense(1024),
            layers.Act('Relu'),
            layers.Norm(),
            layers.Dense(1)
        ]
        return self.sequential(layer_list)

    def call(self, x):
        #         x = self.dense(x)
        x = self.first_norm(x)
        probs = self.classifier(x)
        shaped = self.reshape(x)
        reconstruct = self.decoder(shaped)
        return probs, reconstruct
