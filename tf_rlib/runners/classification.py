import tensorflow as tf
from tf_rlib.models.pyramidnet import PyramidNet
from tf_rlib.runners import runner
from absl import flags
from absl import logging
import numpy as np

FLAGS = flags.FLAGS


class ClassificationRunner(runner.Runner):
    """
    TODO:
        AdamW, Lookahead, MultiGPU, metrics, WeightDecay, mixup
    """
    def __init__(self, train_dataset, valid_dataset=None):
        self.model = PyramidNet()
        self.model(
            tf.keras.Input(shape=train_dataset.element_spec[0].shape[1:],
                           dtype=tf.float32))
        logging.info(self.model.num_params)
        train_metrics = {
            'loss': tf.keras.metrics.MeanTensor('loss'),
            'acc': tf.keras.metrics.SparseCategoricalAccuracy('acc')
        }
        valid_metrics = {
            'loss':
            tf.keras.metrics.SparseCategoricalCrossentropy('loss',
                                                           from_logits=True),
            'acc':
            tf.keras.metrics.SparseCategoricalAccuracy('acc')
        }
        super(ClassificationRunner,
              self).__init__({self.model.name: self.model},
                             train_dataset,
                             valid_dataset=valid_dataset,
                             train_metrics=train_metrics,
                             valid_metrics=valid_metrics,
                             best_state='acc')
        self.criterion = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True)
        self.optim = tf.keras.optimizers.Adam(FLAGS.lr,
                                              beta_1=FLAGS.adam_beta_1,
                                              beta_2=FLAGS.adam_beta_2,
                                              epsilon=FLAGS.adam_epsilon)

    @tf.function
    def train_step(self, x, y):
        """
        Args:
            x (batch): mini batch data
            y (batch): mini batch label
        Returns:
            losses (dict)
        """
        with tf.GradientTape() as tape:
            logits = self.model(x)
            loss = self.criterion(y, logits)

        probs = tf.nn.softmax(logits)
        grads = tape.gradient(loss, self.model.trainable_weights)
        self.optim.apply_gradients(zip(grads, self.model.trainable_weights))
        return {'loss': [loss], 'acc': [y, probs]}

    @tf.function
    def validate_step(self, x, y):
        """
        Args:
            x (batch): mini batch data
            y (batch): mini batch label
        Returns:
            metrics (dict)
        """
        logits = self.model(x)
        probs = tf.nn.softmax(logits)
        return {'loss': [y, logits], 'acc': [y, probs]}

    @tf.function
    def inference(self, dataset):
        raise NotImplementedError
