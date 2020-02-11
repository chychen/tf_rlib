import tensorflow as tf
import tensorflow_addons as tfa
from tf_rlib.models import AE
from tf_rlib.runners.base import runner
from tf_rlib import metrics, losses
from absl import flags
from absl import logging
import numpy as np

FLAGS = flags.FLAGS


class ADAERunner(runner.Runner):
    """ Anomaly Detection Auto Encoder
    """
    LOSSES_POOL = {'mse': losses.MSELoss, 'mae': losses.MAELoss}

    def __init__(self, train_dataset, valid_dataset=None):
        super(ADAERunner, self).__init__(train_dataset,
                                         valid_dataset=valid_dataset,
                                         best_state='loss')

    def init(self):
        self.model = AE()
        train_metrics = {
            'loss': tf.keras.metrics.Mean('loss'),
            'mse': tf.keras.metrics.MeanSquaredError('mse'),
            'mae': tf.keras.metrics.MeanAbsoluteError('mae')
        }
        valid_metrics = {  #TODO ssim, accuracy, AUC
            'loss': tf.keras.metrics.Mean('loss'),
            'mse': tf.keras.metrics.MeanSquaredError('mse'),
            'mae': tf.keras.metrics.MeanAbsoluteError('mae')
        }

        self.loss_object = ADAERunner.LOSSES_POOL[FLAGS.loss_fn]()
        self.optim = tfa.optimizers.AdamW(weight_decay=FLAGS.wd,
                                          lr=0.0,
                                          beta_1=FLAGS.adam_beta_1,
                                          beta_2=FLAGS.adam_beta_2,
                                          epsilon=FLAGS.adam_epsilon)
        return {'ae': self.model}, None, train_metrics, valid_metrics

    def begin_fit_callback(self, lr):
        self.init_lr = lr
        self.lr_scheduler = tf.keras.experimental.CosineDecay(
            self.init_lr, None)

    def begin_epoch_callback(self, epoch_id, epochs):
        if epoch_id < FLAGS.warmup:
            self.optim.learning_rate = epoch_id / FLAGS.warmup * self.init_lr
        else:
            self.lr_scheduler.decay_steps = epochs
            self.optim.learning_rate = self.lr_scheduler(epoch_id)

        self.log_scalar('lr', self.optim.learning_rate, training=True)
        self.optim.weight_decay = FLAGS.wd * self.optim.learning_rate / self.init_lr
        self.log_scalar('wd', self.optim.weight_decay, training=True)

    def train_step(self, x, y):
        """
        Args:
            x (batch): mini batch data
            y (batch): mini batch label
        Returns:
            losses (dict)
        """
        with tf.GradientTape() as tape:
            code, dec_x = self.model(x, training=True)
            loss = self.loss_object(y, dec_x)
            loss = tf.nn.compute_average_loss(
                loss, global_batch_size=FLAGS.bs)  # distributed-aware
            regularization_loss = tf.nn.scale_regularization_loss(
                tf.math.add_n(self.model.losses))  # distributed-aware
            total_loss = loss + regularization_loss

        grads = tape.gradient(total_loss, self.model.trainable_weights)
        self.optim.apply_gradients(zip(grads, self.model.trainable_weights))
        return {'loss': [loss], 'mse': [y, dec_x], 'mae': [y, dec_x]}

    def validate_step(self, x, y):
        """
        Args:
            x (batch): mini batch data
            y (batch): mini batch label
        Returns:
            metrics (dict)
        """
        code, dec_x = self.model(x, training=False)
        loss = self.loss_object(y, dec_x)
        loss = tf.nn.compute_average_loss(
            loss, global_batch_size=FLAGS.bs)  # distributed-aware
        return {'loss': [loss], 'mse': [y, dec_x], 'mae': [y, dec_x]}

    @property
    def required_flags(self):
        return ['dim', 'out_dim', 'bs', 'latent_dim', 'loss_fn']
