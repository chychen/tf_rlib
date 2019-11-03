import tensorflow as tf
from tf_rlib.models import ResNet_Cifar10
from tf_rlib.runners import runner
from tf_rlib.datasets import get_cifar10
from absl import flags
from absl import logging
import numpy as np

FLAGS = flags.FLAGS


class ClassificationResNet18Cifar10(runner.Runner):
    """
    Dataset: Cifar10
    Model: ResNet-18
    Epochs: 300
    Scheduled LR: 1e-1 (1-150), 1e-2 (150-225), 1e-3 (225-300)
    Optimizer: SGD+momentum(0.9)+nesterov
    Accuracy%: 93.6
    Parameters: 11,173,962
    """
    def __init__(self):
        # cifar10
        train_dataset, valid_dataset = get_cifar10()
        # resnet-18
        FLAGS.depth = 18
        self.model = ResNet_Cifar10([2, 2, 2, 2])

        train_metrics = {
            'loss': tf.keras.metrics.MeanTensor('loss'),
            'acc': tf.keras.metrics.SparseCategoricalAccuracy('acc')
        }
        valid_metrics = {
            'loss':
            tf.keras.metrics.SparseCategoricalCrossentropy('loss',
                                                           from_logits=True),
            'acc':
            tf.keras.metrics.SparseCategoricalAccuracy('acc')
        }
        self.loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True)
        self.optim = tf.keras.optimizers.SGD(0.0, 0.9, nesterov=True)

        super(ClassificationResNet18Cifar10,
              self).__init__({'resnet': self.model},
                             train_dataset,
                             valid_dataset=valid_dataset,
                             train_metrics=train_metrics,
                             valid_metrics=valid_metrics,
                             best_state='acc')
        # start training
        self.fit(300, 1e-1)

    def begin_epoch_callback(self, epoch_id, epochs):
        if epoch_id >= 0 and epoch_id < 150:
            self.optim.lr = 1e-1
        elif epoch_id >= 150 and epoch_id < 225:
            self.optim.lr = 1e-2
        elif epoch_id >= 225:
            self.optim.lr = 1e-3

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
        probs = tf.nn.softmax(logits)

        return {'loss': [loss], 'acc': [y, probs]}

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
        probs = tf.nn.softmax(logits)
        return {'loss': [y, logits], 'acc': [y, probs]}

    @tf.function
    def inference(self, dataset):
        raise NotImplementedError
