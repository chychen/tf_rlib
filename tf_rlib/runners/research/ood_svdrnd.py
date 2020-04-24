import tensorflow as tf
from tensorflow.keras.mixed_precision import experimental as mixed_precision
import tensorflow_addons as tfa
import tensorflow_hub as hub
from tf_rlib.models.research import Predictor18 as Predictor, RandomNet18 as RandomNet
from tf_rlib.runners.base import runner
from tf_rlib.datasets import SVDBlurCifar10vsSVHN
from tf_rlib import losses, metrics, layers
from tf_rlib import utils
from absl import flags
from absl import logging
import numpy as np

FLAGS = flags.FLAGS


class OodSvdRndRunner(runner.Runner):
    """ [Novelty Detection Via Blurring, ICLR 2020](https://arxiv.org/abs/1911.11943)
    """
    def __init__(self, train_dataset, valid_dataset=None):
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        super(OodSvdRndRunner, self).__init__(self.train_dataset,
                                              valid_dataset=self.valid_dataset,
                                              best_state='tnr@95tpr')

    def init(self):
        valid_amount = 0
        for x_batch, _ in self.valid_dataset:
            valid_amount = valid_amount + x_batch.shape[0]

        input_shape = self.train_dataset.element_spec[0][0].shape[
            1:]  # [0][0] -> ((x, x_blur),y) -> x
        self.predictor = Predictor()
        self.randnet_1 = RandomNet()  # Note: fixed, not trainable
        self.randnet_2 = RandomNet()  # Note: fixed, not trainable
        self.norm_all = layers.Norm(center=False, scale=False)
        self.randnet_1.trainable = False
        self.randnet_2.trainable = False
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
        if FLAGS.amp:
            self.optim = mixed_precision.LossScaleOptimizer(
                self.optim,
                loss_scale=tf.mixed_precision.experimental.DynamicLossScale(
                    initial_loss_scale=(2**15),
                    increment_period=20,
                    multiplier=2.0))
        return {
            'predictor': self.predictor,
            'randnet_1': self.randnet_1,
            'randnet_2': self.randnet_2
        }, {
            'predictor': input_shape,
            'randnet_1': input_shape,
            'randnet_2': input_shape
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
        ori_x, blur_x = x
        with tf.GradientTape() as tape:
            f1 = self.predictor(ori_x)
            f2 = self.predictor(blur_x)
            g1 = self.randnet_1(ori_x, training=False)
            g2 = self.randnet_2(blur_x, training=False)
            all_ = self.norm_all(tf.concat([f1, f2, g1, g2], axis=0),
                                 training=True)
            f1, f2, g1, g2 = tf.split(all_, [
                FLAGS.bs,
            ] * 4, axis=0)

            if FLAGS.amp:
                f1 = tf.cast(f1, tf.float32)
                f2 = tf.cast(f2, tf.float32)
                g1 = tf.cast(g1, tf.float32)
                g2 = tf.cast(g2, tf.float32)

            loss1 = self.loss_object(g1, f1)
            loss2 = self.loss_object(g2, f2)
            loss = (loss1 + loss2) / 2.0
            # distributed-aware
            loss = tf.nn.compute_average_loss(loss, global_batch_size=FLAGS.bs)
            regularization_loss = tf.nn.scale_regularization_loss(
                tf.math.add_n(self.predictor.losses))
            if FLAGS.amp:
                total_loss = loss + regularization_loss
                total_loss = self.optim.get_scaled_loss(total_loss)
            else:
                total_loss = loss + regularization_loss

        trainable_w = []
        for m in [
                self.predictor,
        ]:
            trainable_w = trainable_w + m.trainable_weights
        if FLAGS.amp:
            grads = tape.gradient(total_loss, trainable_w)
            grads = self.optim.get_unscaled_gradients(grads)
        else:
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
        f = self.norm_all(self.predictor(x, training=False), training=False)
        g = self.norm_all(self.randnet_1(x, training=False), training=False)
        #         f = self.predictor(x, training=False)
        #         g = self.randnet_1(x, training=False)
        loss = self.loss_object(g, f)
        return {
            'figure': [y, loss[..., None]],
            'tnr@95tpr': [y, loss[..., None]]
        }

    def custom_log_data(self, x_batch, y_batch):
        if type(x_batch) is tuple:  # training
            ori_x, blur_x = x_batch
            return {'ori_x': ori_x, 'blur_x': blur_x}
        else:  # validation
            return {'ori_x': x_batch}

    @property
    def required_flags(self):
        return ['dim', 'bs']

    @property
    def support_amp(self):
        return True
