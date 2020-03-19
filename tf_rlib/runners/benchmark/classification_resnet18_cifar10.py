import tensorflow as tf
from tensorflow.keras.mixed_precision import experimental as mixed_precision
from tf_rlib.models import ResNet18
from tf_rlib.runners.base import runner
# from tf_rlib.datasets import Cifar10RandAugment as Cifar10
from tf_rlib.datasets import Cifar10
from tf_rlib import utils
from absl import flags
from absl import logging
import numpy as np

FLAGS = flags.FLAGS


class ResNet18Cifar10(runner.Runner):
    """
    Dataset: Cifar10
    Model: ResNet-18
    Epochs: 300
    Scheduled LR: 1e-1 (1-150), 1e-2 (150-225), 1e-3 (225-300)
    Optimizer: SGD+momentum(0.9)+nesterov
    """
    def __init__(self, preact, last_norm):
        self.preact = preact
        self.last_norm = last_norm
        utils.set_gpus('1')
        # resnet-18
        FLAGS.dim = 2
        FLAGS.out_dim = 10
        FLAGS.bs = 128
        FLAGS.l2 = 5e-4
        FLAGS.depth = 18
        # cifar10
        train_dataset, valid_dataset = Cifar10().get_data()
        super(ResNet18Cifar10, self).__init__(train_dataset,
                                              valid_dataset=valid_dataset,
                                              best_state='acc')
        # start training
        self.fit(300, 1e-1)

    def init(self):
        self.model = ResNet18(feature_mode=False,
                              preact=self.preact,
                              last_norm=self.last_norm)
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
        self.optim = tf.keras.optimizers.SGD(0.0, 0.9, nesterov=False)
        if FLAGS.amp:
            self.optim = mixed_precision.LossScaleOptimizer(
                self.optim, loss_scale='dynamic')
        return {'resnet': self.model}, None, train_metrics, valid_metrics

    def begin_epoch_callback(self, epoch_id, epochs):
        if epoch_id >= 0 and epoch_id < 150:
            self.optim.lr = 1e-1
        elif epoch_id >= 150 and epoch_id < 225:
            self.optim.lr = 1e-2
        elif epoch_id >= 225:
            self.optim.lr = 1e-3

        self.log_scalar('lr', self.optim.lr, training=True)

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


class ClassificationResNet18Cifar10(ResNet18Cifar10):
    """
    Accuracy%: 94.31
    Accuracy%: 94.44 (with RandAugment)
    Parameters: 11,178,186
    """
    def __init__(self):
        super(ClassificationResNet18Cifar10, self).__init__(preact=False,
                                                            last_norm=False)


class ClassificationResNet18PreactCifar10(ResNet18Cifar10):
    """
    Accuracy%: 95.00
    Parameters: 11,176,272
    """
    def __init__(self):
        super(ClassificationResNet18PreactCifar10,
              self).__init__(preact=True, last_norm=False)


class ClassificationResNet18PreactLastnormCifar10(ResNet18Cifar10):
    """
    Accuracy%: 94.??
    Parameters: 11,180,112
    """
    def __init__(self):
        super(ClassificationResNet18PreactLastnormCifar10,
              self).__init__(preact=True, last_norm=True)
