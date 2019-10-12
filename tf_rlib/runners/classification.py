import tensorflow as tf
from tf_rlib.models.pyramidnet import PyramidNet
from tf_rlib.runners import runner
from absl import flags
from absl import logging
import numpy as np

FLAGS = flags.FLAGS


class ClassificationRunner(runner.Runner):
    """
    TODO:
        AdamW, Lookahead, MultiGPU, WeightDecay, mixup
    """
    def __init__(self, train_dataset, valid_dataset=None):
        self.model = PyramidNet()
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
        self.criterion = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True)
        self.lr_scheduler = tf.keras.experimental.CosineDecay(FLAGS.lr, None)
        self.optim = tf.keras.optimizers.Adam(FLAGS.lr,
                                              beta_1=FLAGS.adam_beta_1,
                                              beta_2=FLAGS.adam_beta_2,
                                              epsilon=FLAGS.adam_epsilon)
        #         if FLAGS.amp:
        #             self.optim = tf.keras.mixed_precision.experimental.LossScaleOptimizer(self.optim, "dynamic")

        super(ClassificationRunner, self).__init__({'pyramidnet': self.model},
                                                   {'adam': self.optim},
                                                   train_dataset,
                                                   valid_dataset=valid_dataset,
                                                   train_metrics=train_metrics,
                                                   valid_metrics=valid_metrics,
                                                   best_state='acc')

    def set_epoch_lr_callback(self, epoch_id, epochs):
        self.lr_scheduler.decay_steps = epochs
        self.optim.lr = self.lr_scheduler(epoch_id)

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
            loss = self.criterion(y, logits)

        grads = tape.gradient(loss, self.model.trainable_weights)
        self.optim.apply_gradients(zip(grads, self.model.trainable_weights))
        probs = tf.nn.softmax(logits)

        return {'loss': [loss], 'acc': [y, probs]}


#             if FLAGS.amp:
#                 scaled_loss = self.optim.get_scaled_loss(loss)
#                 scaled_grads = tape.gradient(scaled_loss, self.model.trainable_weights)
#                 grads = self.optim.get_unscaled_gradients(scaled_grads)
#             else:
#                 grads = tape.gradient(loss, self.model.trainable_weights)

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
