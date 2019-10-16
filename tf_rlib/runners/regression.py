import tensorflow as tf
from tf_rlib.models.pyramidnet import PyramidNet
from tf_rlib.runners import runner
from absl import flags
from absl import logging
import numpy as np

FLAGS = flags.FLAGS


class RegressionRunner(runner.Runner):
    def __init__(self, train_dataset, valid_dataset=None):
        self.model = PyramidNet()
        train_metrics = {
            'loss': tf.keras.metrics.MeanTensor('loss'),
            'mse': tf.keras.metrics.MeanSquaredError('mse'),
            'mape': tf.keras.metrics.MeanAbsolutePercentageError('mape'),
            'mae': tf.keras.metrics.MeanAbsoluteError('mae')
        }
        valid_metrics = {
            'loss': tf.keras.metrics.MeanAbsoluteError('loss'),
            'mse': tf.keras.metrics.MeanSquaredError('mse'),
            'mape': tf.keras.metrics.MeanAbsolutePercentageError('mape'),
            'mae': tf.keras.metrics.MeanAbsoluteError('mae')
        }
        self.loss_fn = tf.keras.losses.MAE()
        #         self.loss_fn = tf.keras.losses.MSE()
        self.optim = tf.keras.optimizers.SGD(0.0, 0.9)

        super(RegressionRunner, self).__init__({'pyramidnet': self.model},
                                               train_dataset,
                                               valid_dataset=valid_dataset,
                                               train_metrics=train_metrics,
                                               valid_metrics=valid_metrics,
                                               best_state='mae')

    def begin_fit_callback(self, lr):
        self.init_lr = lr
        self.lr_scheduler = tf.keras.experimental.CosineDecay(
            self.init_lr, None)
        self.optim.lr = self.init_lr

    def begin_epoch_callback(self, epoch_id, epochs):
        if epoch_id < FLAGS.warmup:
            self.optim.lr = epoch_id / FLAGS.warmup * self.init_lr
        else:
            self.lr_scheduler.decay_steps = epochs
            self.optim.lr = self.lr_scheduler(epoch_id)

        self.log_scalar('lr', self.optim.lr, epoch_id, training=True)

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
            logits = self.model(x, training=True)
            loss = self.loss_fn(y, logits)
            regularization_loss = tf.math.add_n(self.model.losses)
            total_loss = loss + regularization_loss

        grads = tape.gradient(total_loss, self.model.trainable_weights)
        self.optim.apply_gradients(zip(grads, self.model.trainable_weights))

        return {
            'loss': [loss],
            'mse': [y, logits],
            'mape': [y, logits],
            'mae': [y, logits]
        }

    @tf.function
    def validate_step(self, x, y):
        """
        Args:
            x (batch): mini batch data
            y (batch): mini batch label
        Returns:
            metrics (dict)
        """
        logits = self.model(x, training=False)
        return {
            'loss': [loss],
            'mse': [y, logits],
            'mape': [y, logits],
            'mae': [y, logits]
        }

    @tf.function
    def inference(self, dataset):
        raise NotImplementedError
