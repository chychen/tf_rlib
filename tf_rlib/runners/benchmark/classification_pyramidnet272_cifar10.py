import tensorflow as tf
from tf_rlib.models import PyramidNet
from tf_rlib.runners import runner
from tf_rlib.datasets import Cifar10
from absl import flags
from absl import logging
import numpy as np

FLAGS = flags.FLAGS


class ClassificationPyramidNet272Cifar10(runner.Runner):
    """ 4 gpus training
    Dataset: Cifar10
    Model: PyramidNet272
    Epochs: 300
    Scheduled LR: 1e-1 (1-150), 1e-2 (150-225), 1e-3 (225-300)
    Optimizer: SGD+momentum(0.9)+nesterov
    Accuracy%: 96.66%
    Parameters: 26,049,562
    """
    def __init__(self):
        FLAGS.gpus = '0,1,2,3'
        FLAGS.exp_name = ClassificationPyramidNet272Cifar10.__name__
        # pyramidnet-272
        FLAGS.l2 = 1e-4
        FLAGS.depth = 272
        FLAGS.model_alpha = 200
        FLAGS.bottleneck = True
        # cifar10
        train_dataset, valid_dataset = Cifar10().get_data()
        super(ClassificationPyramidNet272Cifar10,
              self).__init__(train_dataset,
                             valid_dataset=valid_dataset,
                             best_state='acc')
        # start training
        self.fit(300, 1e-1)

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
        self.optim = tf.keras.optimizers.SGD(0.0, 0.9, nesterov=True)
        return {'pyramidnet': self.model}, train_metrics, valid_metrics

    def begin_epoch_callback(self, epoch_id, epochs):
        if epoch_id >= 0 and epoch_id < 150:
            self.optim.lr = 1e-1
        elif epoch_id >= 150 and epoch_id < 225:
            self.optim.lr = 1e-2
        elif epoch_id >= 225:
            self.optim.lr = 1e-3

        self.log_scalar('lr', self.optim.lr, epoch_id, training=True)

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
