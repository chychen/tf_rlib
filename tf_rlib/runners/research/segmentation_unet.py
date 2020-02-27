import tensorflow as tf
import tensorflow_addons as tfa
from tf_rlib.models import UNet as net
from tf_rlib.runners.base import runner
from tf_rlib.metrics import DiceCoefficient
from tf_rlib.losses import CustomLoss
from tf_rlib.losses.dice import dice_loss
from absl import flags
from absl import logging
import numpy as np

FLAGS = flags.FLAGS


def loss(y_true, y_pred, smooth=1):
    dice = dice_loss(y_true, y_pred, smooth=1)
    ce = tf.keras.backend.mean(
        tf.keras.losses.categorical_crossentropy(y_true, y_pred),
        axis=tuple(i + 1 for i in range(tf.shape(y_true).shape[0] - 2)))
    return dice + ce


class SegmentationRunner(runner.Runner):
    def __init__(self, train_dataset, valid_dataset=None):
        super(SegmentationRunner, self).__init__(train_dataset,
                                                 valid_dataset=valid_dataset,
                                                 best_state='dice_coef')

    def init(self):
        self.model = net()
        train_metrics = {
            'loss': tf.keras.metrics.Mean('loss'),
            'dice_coef': DiceCoefficient()
        }
        valid_metrics = {
            'loss': tf.keras.metrics.Mean('loss'),
            'dice_coef': DiceCoefficient()
        }
        self.loss_object = CustomLoss(loss,
                                      reduction=tf.keras.losses.Reduction.NONE)
        self.optim = tfa.optimizers.AdamW(weight_decay=FLAGS.wd,
                                          lr=0.0,
                                          beta_1=FLAGS.adam_beta_1,
                                          beta_2=FLAGS.adam_beta_2,
                                          epsilon=FLAGS.adam_epsilon)
        return {'unet': self.model}, None, train_metrics, valid_metrics

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
            logits = self.model(x, training=True)
            loss = self.loss_object(y, logits)
            loss = tf.nn.compute_average_loss(
                loss, global_batch_size=FLAGS.bs)  # distributed-aware
            regularization_loss = tf.nn.scale_regularization_loss(
                tf.math.add_n(self.model.losses))  # distributed-aware
            total_loss = loss + regularization_loss

        grads = tape.gradient(total_loss, self.model.trainable_weights)
        self.optim.apply_gradients(zip(grads, self.model.trainable_weights))
        probs = tf.nn.softmax(logits)
        return {'loss': [loss], 'dice_coef': [y, probs]}

    def validate_step(self, x, y):
        """
        Args:
            x (batch): mini batch data
            y (batch): mini batch label
        Returns:
            metrics (dict)
        """
        logits = self.model(x, training=False)
        loss = self.loss_object(y, logits)
        loss = tf.nn.compute_average_loss(loss, global_batch_size=FLAGS.bs)
        probs = tf.nn.softmax(logits)
        return {'loss': [loss], 'dice_coef': [y, probs]}

    @property
    def required_flags(self):
        return ['dim', 'out_dim', 'bs']

    @property
    def support_amp(self):
        return False

    @tf.function
    def inference(self, x):
        bs = FLAGS.bs
        iters = int(len(x) / bs)
        remain = len(x) % bs
        results = []
        for i in range(iters):
            data = x[i * bs:(i + 1) * bs]
            result = self.model(data, training=False)
            results.append(result)
        if remain:
            data = x[i * bs:(i + 1) * bs]
            result = self.model(data, training=False)
            results.append(result)
        return tf.concat(results, axis=0)
