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
        AdamW, Lookahead, MultiGPU, tensorboard, metrics, WeightDecay
    """
    def __init__(self, train_dataset, valid_dataset=None):
        self.model = PyramidNet()
        self.model(
            tf.keras.Input(shape=train_dataset.element_spec[0].shape[1:],
                           dtype=tf.float32))
        logging.info(self.model.num_params)
        metrics = {
            'train_loss':
            tf.keras.metrics.MeanTensor(),
            'valid_loss':
            tf.keras.metrics.SparseCategoricalCrossentropy(from_logits=True),
            'valid_acc':
            tf.keras.metrics.SparseCategoricalAccuracy()
        }
        super(ClassificationRunner, self).__init__({'Classifier': self.model},
                                                   train_dataset,
                                                   valid_dataset=valid_dataset,
                                                   metrics=metrics,
                                                   best_state='valid_acc')
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

        grads = tape.gradient(loss, self.model.trainable_weights)
        self.optim.apply_gradients(zip(grads, self.model.trainable_weights))
        return {'train_loss': [loss]}

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
        return {'valid_loss': [y, logits], 'valid_acc': [y, probs]}

    def inference(self, dataset):
        raise NotImplementedError
