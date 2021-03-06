import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow_hub as hub
from tf_rlib.models.research import Predictor34 as Predictor, RandomNet34 as RandomNet
from tf_rlib.runners.base import runner
from tf_rlib import losses, metrics
from absl import flags
from absl import logging
import numpy as np

FLAGS = flags.FLAGS


class OodRndRunner(runner.Runner):
    """
    """
    def __init__(self, train_dataset, valid_dataset=None):
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        super(OodRndRunner, self).__init__(train_dataset,
                                           valid_dataset=valid_dataset,
                                           best_state='tnr@95tpr')

    def init(self):
        valid_amount = 0
        for x_batch, _ in self.valid_dataset:
            valid_amount = valid_amount + x_batch.shape[0]

        input_shape = self.train_dataset.element_spec[0].shape[1:]
        self.predictor = Predictor()
        self.randnet = RandomNet()  # Note: fixed, not trainable
        self.randnet.trainable = False

        train_metrics = {
            'loss': tf.keras.metrics.Mean('loss'),
        }
        valid_metrics = {
            'figure':
            metrics.RNDMetrics(mode=metrics.RNDMetrics.MODE_FIGURE,
                               amount=valid_amount),
            'tnr@95tpr':
            metrics.RNDMetrics(mode=metrics.RNDMetrics.MODE_TNR95TPR,
                               amount=valid_amount)
        }
        self.loss_object = losses.MSELoss()
        self.optim = tfa.optimizers.AdamW(weight_decay=FLAGS.wd,
                                          lr=0.0,
                                          beta_1=FLAGS.adam_beta_1,
                                          beta_2=FLAGS.adam_beta_2,
                                          epsilon=FLAGS.adam_epsilon)
        return {
            'predictor': self.predictor,
            'randnet': self.randnet
        }, {
            'predictor': input_shape,
            'randnet': input_shape
        }, train_metrics, valid_metrics

    # TODO: wrap up as one cosineannealing api
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
            f = self.predictor(x)
            g = self.randnet(x, training=False)
            loss = self.loss_object(g, f)
            # distributed-aware
            loss = tf.nn.compute_average_loss(loss, global_batch_size=FLAGS.bs)
            regularization_loss = tf.nn.scale_regularization_loss(
                tf.math.add_n(self.predictor.losses))
            total_loss = loss + regularization_loss

        trainable_w = self.predictor.trainable_weights
        grads = tape.gradient(total_loss, trainable_w)
        self.optim.apply_gradients(zip(grads, trainable_w))
        return {'loss': [loss]}

    def validate_step(self, x, y):
        """
        Args:
            x (batch): mini batch data
            y (batch): mini batch label
        Returns:
            metrics (dict)
        """
        f = self.predictor(x, training=False)
        g = self.randnet(x, training=False)
        loss = self.loss_object(g, f)
        # TODO distributed-aware
        #         loss = tf.nn.compute_average_loss(loss, global_batch_size=FLAGS.bs)
        return {
            'figure': [y, loss[..., None]],
            'tnr@95tpr': [y, loss[..., None]]
        }

    def custom_log_data(self, x_batch, y_batch):
        return None

    @property
    def required_flags(self):
        return ['dim', 'bs']

    @property
    def support_amp(self):
        return False
