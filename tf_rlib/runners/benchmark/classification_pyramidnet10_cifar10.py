import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.mixed_precision import experimental as mixed_precision
from tf_rlib.models import PyramidNet
from tf_rlib.runners.base import runner
from tf_rlib.datasets import Cifar10
from absl import flags
from absl import logging
import numpy as np

FLAGS = flags.FLAGS


class ClassificationPyramidNet10Cifar10(runner.Runner):
    """ 4 gpus training
    Dataset: Cifar10
    Model: PyramidNet10
    Epochs: 100
    Optimizer: AdamW + WeightDecay=0.0
    Parameters: 291,482
    
    (Float32)
        Batch Size: 128
        Scheduled LR: 1e-3 + CosineAnealing
        Accuracy%: 87.6%~88.3%
    (Float16, speedup about 10%~15% than float32 with such small network)
        Batch Size: 256
        Scheduled LR: 2e-3 + CosineAnealing
        Accuracy%: 88.5%
    """
    def __init__(self):
        FLAGS.gpus = '0'
        # pyramidnet-10
        amp_factor = 2 if FLAGS.amp else 1
        FLAGS.bs = 128 * amp_factor
        FLAGS.dim = 2
        FLAGS.out_dim = 10
        FLAGS.depth = 10
        FLAGS.bottleneck = False
        layers_per_block = 3 if FLAGS.bottleneck else 2
        total_blocks = (FLAGS.depth - 2) / layers_per_block
        FLAGS.model_alpha = total_blocks * 8
        FLAGS.lr = 1e-3 * amp_factor
        FLAGS.epochs = 100
        # cifar10
        train_dataset, valid_dataset = Cifar10().get_data()
        super(ClassificationPyramidNet10Cifar10,
              self).__init__(train_dataset,
                             valid_dataset=valid_dataset,
                             best_state='acc')
        # start training
        self.fit(FLAGS.epochs, 1e-1)

    def init(self):
        self.model = PyramidNet()
        train_metrics = {
            'loss': tf.keras.metrics.Mean('loss'),
            'acc': tf.keras.metrics.SparseCategoricalAccuracy('acc')
        }
        valid_metrics = {
            'loss': tf.keras.metrics.Mean('loss'),
            'acc': tf.keras.metrics.SparseCategoricalAccuracy('acc')
        }
        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True,
            reduction=tf.keras.losses.Reduction.NONE)  # distributed-aware
        self.optim = tfa.optimizers.AdamW(weight_decay=FLAGS.wd,
                                          lr=0.0,
                                          beta_1=FLAGS.adam_beta_1,
                                          beta_2=FLAGS.adam_beta_2,
                                          epsilon=FLAGS.adam_epsilon)
        if FLAGS.amp:
            self.optim = mixed_precision.LossScaleOptimizer(
                self.optim, loss_scale='dynamic')
        return {'pyramidnet': self.model}, train_metrics, valid_metrics

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

        self.log_scalar('lr',
                        self.optim.learning_rate,
                        epoch_id,
                        training=True)
        self.optim.weight_decay = FLAGS.wd * self.optim.learning_rate / self.init_lr
        self.log_scalar('wd', self.optim.weight_decay, epoch_id, training=True)

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
            if FLAGS.amp:
                regularization_loss = tf.cast(regularization_loss, tf.float16)
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
        probs = tf.nn.softmax(logits)
        return {'loss': [loss], 'acc': [y, probs]}

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
        return {'loss': [loss], 'acc': [y, probs]}

    @tf.function
    def inference(self, dataset):
        raise NotImplementedError

    @property
    def required_flags(self):
        pass

    @property
    def support_amp(self):
        return True
