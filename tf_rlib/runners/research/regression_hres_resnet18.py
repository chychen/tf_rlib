import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.mixed_precision import experimental as mixed_precision
from tf_rlib.models.research import HResResNet18
from tf_rlib.runners.base import runner
from tf_rlib import metrics
from tf_rlib import losses
from absl import flags
from absl import logging
import numpy as np

FLAGS = flags.FLAGS
LOGGER = logging.get_absl_logger()

SUB_Y = [2, 9]


class RegHResResNet18Runner(runner.Runner):
    """ 
    Regression High Resolution ResNet18 Runner
    """
    def __init__(self, train_dataset, valid_dataset=None, y_denorm_fn=None):
        self.y_denorm_fn = y_denorm_fn
        super(RegHResResNet18Runner,
              self).__init__(train_dataset,
                             valid_dataset=valid_dataset,
                             best_state='rmse')

    def init(self):
        self.model = HResResNet18(preact=True)
        train_metrics = {
            'loss':
            tf.keras.metrics.MeanTensor('loss'),
            'rmse':
            metrics.RootMeanSquaredError('rmse', denorm_fn=self.y_denorm_fn),
            'mae':
            metrics.MeanAbsoluteError('mae', denorm_fn=self.y_denorm_fn)
        }
        valid_metrics = {
            'loss':
            tf.keras.metrics.MeanTensor('loss'),
            'rmse':
            metrics.RootMeanSquaredError('rmse', denorm_fn=self.y_denorm_fn),
            'mae':
            metrics.MeanAbsoluteError('mae', denorm_fn=self.y_denorm_fn)
        }
        self.optim = tfa.optimizers.AdamW(weight_decay=FLAGS.wd,
                                          lr=0.0,
                                          beta_1=FLAGS.adam_beta_1,
                                          beta_2=FLAGS.adam_beta_2,
                                          epsilon=FLAGS.adam_epsilon)
        if FLAGS.amp:
            self.optim = mixed_precision.LossScaleOptimizer(
                self.optim, loss_scale='dynamic')
        return {'HResResNet18': self.model}, None, train_metrics, valid_metrics

    def begin_fit_callback(self, lr):
        self.init_lr = lr
        self.lr_scheduler = tf.keras.experimental.CosineDecay(
            self.init_lr, None)

    def begin_epoch_callback(self, epoch_id, epochs):
        self.optim.weight_decay = 0.0
        self.log_scalar('lr', self.optim.learning_rate, training=True)
        self.log_scalar('wd', self.optim.weight_decay, training=True)

    def begin_step_callback(self, step_id, epochs):
        if step_id < FLAGS.steps_per_epoch * FLAGS.warmup:
            self.optim.learning_rate = step_id / (FLAGS.steps_per_epoch *
                                                  FLAGS.warmup) * self.init_lr
        else:
            self.lr_scheduler.decay_steps = FLAGS.steps_per_epoch * epochs
            self.optim.learning_rate = self.lr_scheduler(step_id)
        self.optim.weight_decay = FLAGS.wd * self.optim.learning_rate / self.init_lr

    def train_step(self, x, y):
        """
        """
        m = tf.cast(x[..., 16:], tf.bool)
        m = tf.gather(m, SUB_Y, axis=-1)
        m = tf.logical_not(m)
        m = tf.concat([m, m], axis=-1)
        with tf.GradientTape() as tape:
            logits = self.model(x, training=True)
            if FLAGS.loss_fn == 'mse':
                loss = self.mse_mask(y, logits, m)
            elif FLAGS.loss_fn == 'mae':
                loss = self.mae_mask(y, logits, m)
            else:
                raise NotImplementedError
            loss = tf.nn.compute_average_loss(
                loss, global_batch_size=FLAGS.bs)  # distributed-aware
            regularization_loss = tf.nn.scale_regularization_loss(
                tf.math.add_n(self.model.losses))  # distributed-aware
            if FLAGS.amp:
                regularization_loss = tf.cast(regularization_loss, tf.float16)
                loss = tf.cast(loss, tf.float16)
                total_loss = loss + regularization_loss
                total_loss = self.optim.get_scaled_loss(total_loss)
            else:
                total_loss = loss + regularization_loss

        if FLAGS.amp:
            grads = tape.gradient(total_loss, self.model.trainable_weights)
            grads = self.optim.get_unscaled_gradients(grads)
        else:
            grads = tape.gradient(total_loss, self.model.trainable_weights)

        self.optim.apply_gradients(zip(grads, self.model.trainable_weights))
        return {'loss': [loss], 'rmse': [y, logits], 'mae': [y, logits]}

    def validate_step(self, x, y):
        """
        Args:
            x (batch): mini batch data
            y (batch): mini batch label
        Returns:
            metrics (dict)
        """

        m = tf.cast(x[..., 16:], tf.bool)
        m = tf.gather(m, SUB_Y, axis=-1)
        m = tf.logical_not(m)
        m = tf.concat([m, m], axis=-1)
        logits = self.model(x, training=False)
        if FLAGS.loss_fn == 'mse':
            loss = self.mse_mask(y, logits, m)
        elif FLAGS.loss_fn == 'mae':
            loss = self.mae_mask(y, logits, m)
        else:
            raise NotImplementedError
        loss = tf.nn.compute_average_loss(
            loss, global_batch_size=FLAGS.bs)  # distributed-aware
        return {'loss': [loss], 'rmse': [y, logits], 'mae': [y, logits]}

    @tf.function
    def inference(self, x):
        mask = tf.logical_not(x[..., 16:])
        logits = self.model(x, training=False) * mask
        return logits

    @property
    def required_flags(self):
        return ['dim', 'out_dim', 'bs', 'loss_fn', 'steps_per_epoch']

    @property
    def support_amp(self):
        return True

    def custom_log_data(self, x_batch, y_batch):
        logits = self.model(x_batch, training=False)
        layers = SUB_Y  # 1250m and 4750m
        ret_dict = {}
        for i, x_idx in enumerate(layers):
            ret_dict[f'x_{x_idx}'] = tf.gather(x_batch, [
                2 + x_idx,
            ], axis=-1)
            ret_dict[f'y_{x_idx}'] = tf.gather(y_batch, [i, 2 + i], axis=-1)
            ret_dict[f'out_{x_idx}'] = tf.gather(logits, [i, 2 + i], axis=-1)
        return ret_dict

    def mse_mask(self, y, logits, m):
        return tf.reduce_mean(tf.ragged.boolean_mask((y - logits)**2, m),
                              axis=(1, 2, 3))

    def mae_mask(self, y, logits, m):
        return tf.reduce_mean(tf.ragged.boolean_mask(tf.abs(y - logits), m),
                              axis=(1, 2, 3))
