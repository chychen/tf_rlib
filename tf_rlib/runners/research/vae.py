import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow_hub as hub
from tf_rlib.models.research import VAEEncoder32x32, VAEDecoder32x32, VAEEncoder28x28, VAEDecoder28x28
from tf_rlib.runners.base import runner
from tf_rlib import losses
from absl import flags
from absl import logging
import numpy as np

FLAGS = flags.FLAGS


class VAERunner32x32(runner.Runner):
    """
    """

    LOSSES_POOL = {
        'mse': losses.MSELoss,
        'mae': losses.MAELoss,
        'vae': losses.VAEGaussLoss
    }

    def __init__(self, train_dataset, valid_dataset=None):
        self.train_dataset = train_dataset
        if FLAGS.loss_fn is None:
            FLAGS.loss_fn = 'vae'
        super(VAERunner32x32, self).__init__(train_dataset,
                                             valid_dataset=valid_dataset,
                                             best_state='loss')

    def init(self):
        input_shape = self.train_dataset.element_spec[0].shape[1:]
        self.encoder = VAEEncoder32x32()
        self.decoder = VAEDecoder32x32()
        train_metrics = {
            'loss': tf.keras.metrics.Mean('loss'),
        }
        valid_metrics = {
            'loss': tf.keras.metrics.Mean('loss'),
        }
        self.loss_object = VAERunner32x32.LOSSES_POOL[FLAGS.loss_fn]()
        self.optim = tfa.optimizers.AdamW(weight_decay=FLAGS.wd,
                                          lr=0.0,
                                          beta_1=FLAGS.adam_beta_1,
                                          beta_2=FLAGS.adam_beta_2,
                                          epsilon=FLAGS.adam_epsilon)
        return {
            'encoder': self.encoder,
            'decoder': self.decoder
        }, {
            'encoder': input_shape,
            'decoder': FLAGS.latent_dim
        }, train_metrics, valid_metrics

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
            mean, logvar, z = self.encoder(x)
            if FLAGS.loss_fn == 'vae':
                x_logit = self.decoder(z)
                loss = self.loss_object(x, {
                    'z': z,
                    'x_logit': x_logit,
                    'mean': mean,
                    'logvar': logvar
                })
            else:
                x_logit = self.decoder(mean)
                loss = self.loss_object(x, x_logit)

            # distributed-aware
            loss = tf.nn.compute_average_loss(loss, global_batch_size=FLAGS.bs)
            regularization_loss = tf.nn.scale_regularization_loss(
                tf.math.add_n(self.encoder.losses + self.decoder.losses))
            total_loss = loss + regularization_loss

        trainable_w = self.encoder.trainable_weights + self.decoder.trainable_weights
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
        mean, logvar, z = self.encoder(x, training=False)
        if FLAGS.loss_fn == 'vae':
            x_logit = self.decoder(z, training=False)
            loss = self.loss_object(x, {
                'z': z,
                'x_logit': x_logit,
                'mean': mean,
                'logvar': logvar
            })
        else:
            x_logit = self.decoder(mean, training=False)
            loss = self.loss_object(x, x_logit)
        # distributed-aware
        loss = tf.nn.compute_average_loss(loss, global_batch_size=FLAGS.bs)
        return {'loss': [loss]}

    def custom_log_data(self, x_batch, y_batch):
        mean, logvar, z = self.encoder(x_batch, training=False)
        if FLAGS.loss_fn == 'vae':
            x_logit = self.decoder(z, training=False)
        else:
            x_logit = self.decoder(mean, training=False)

        x_logit = tf.sigmoid(x_logit)
        return {'reconstructs': x_logit}

    @property
    def required_flags(self):
        return ['dim', 'bs', 'out_dim', 'latent_dim']

    @property
    def support_amp(self):
        return False

    @tf.function
    def sample(self, eps=None):
        if eps is None:
            eps = tf.random.normal(shape=(100, self.latent_dim))
        logits = self.decoder(eps, training=False)
        probs = tf.sigmoid(logits)
        return probs


class VAERunner28x28(runner.Runner):
    """
    """

    LOSSES_POOL = {
        'mse': losses.MSELoss,
        'mae': losses.MAELoss,
        'vae': losses.VAEBernoulliLoss
    }

    def __init__(self, train_dataset, valid_dataset=None):
        self.train_dataset = train_dataset
        if FLAGS.loss_fn is None:
            FLAGS.loss_fn = 'vae'
        super(VAERunner28x28, self).__init__(train_dataset,
                                             valid_dataset=valid_dataset,
                                             best_state='loss')

    def init(self):
        input_shape = self.train_dataset.element_spec[0].shape[1:]
        self.encoder = VAEEncoder28x28()
        self.decoder = VAEDecoder28x28()
        train_metrics = {
            'loss': tf.keras.metrics.Mean('loss'),
        }
        valid_metrics = {
            'loss': tf.keras.metrics.Mean('loss'),
        }
        self.loss_object = VAERunner28x28.LOSSES_POOL[FLAGS.loss_fn]()
        self.optim = tfa.optimizers.AdamW(weight_decay=FLAGS.wd,
                                          lr=0.0,
                                          beta_1=FLAGS.adam_beta_1,
                                          beta_2=FLAGS.adam_beta_2,
                                          epsilon=FLAGS.adam_epsilon)
        return {
            'encoder': self.encoder,
            'decoder': self.decoder
        }, {
            'encoder': input_shape,
            'decoder': FLAGS.latent_dim
        }, train_metrics, valid_metrics

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
            mean, logvar, z = self.encoder(x)
            if FLAGS.loss_fn == 'vae':
                x_logit = self.decoder(z)
                loss = self.loss_object(x, {
                    'z': z,
                    'x_logit': x_logit,
                    'mean': mean,
                    'logvar': logvar
                })
            else:
                x_logit = self.decoder(mean)
                loss = self.loss_object(x, x_logit)

            # distributed-aware
            loss = tf.nn.compute_average_loss(loss, global_batch_size=FLAGS.bs)
            regularization_loss = tf.nn.scale_regularization_loss(
                tf.math.add_n(self.encoder.losses + self.decoder.losses))
            total_loss = loss + regularization_loss

        trainable_w = self.encoder.trainable_weights + self.decoder.trainable_weights
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
        mean, logvar, z = self.encoder(x, training=False)
        if FLAGS.loss_fn == 'vae':
            x_logit = self.decoder(z, training=False)
            loss = self.loss_object(x, {
                'z': z,
                'x_logit': x_logit,
                'mean': mean,
                'logvar': logvar
            })
        else:
            x_logit = self.decoder(mean, training=False)
            loss = self.loss_object(x, x_logit)
        # distributed-aware
        loss = tf.nn.compute_average_loss(loss, global_batch_size=FLAGS.bs)
        return {'loss': [loss]}

    def custom_log_data(self, x_batch, y_batch):
        mean, logvar, z = self.encoder(x_batch, training=False)
        if FLAGS.loss_fn == 'vae':
            x_logit = self.decoder(z, training=False)
        else:
            x_logit = self.decoder(mean, training=False)

        x_logit = tf.sigmoid(x_logit)
        return {'reconstructs': x_logit}

    @property
    def required_flags(self):
        return ['dim', 'bs', 'out_dim', 'latent_dim']

    @property
    def support_amp(self):
        return False

    @tf.function
    def sample(self, eps=None):
        if eps is None:
            eps = tf.random.normal(shape=(100, self.latent_dim))
        logits = self.decoder(eps, training=False)
        probs = tf.sigmoid(logits)
        return probs
