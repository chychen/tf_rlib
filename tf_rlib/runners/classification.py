""" Features
- [v] PyramidNet
- [ ] Warmup LR
- [ ] CosineAnnealing LR
- [ ] Mixup
- [ ] AdamW
- [ ] Lookahead
- [v] WeightDecay
"""
import tensorflow as tf
from tf_rlib.models.pyramidnet import PyramidNet
from tf_rlib.runners import runner
from absl import flags
from absl import logging
import numpy as np

FLAGS = flags.FLAGS


class ClassificationRunner(runner.Runner):

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
        self.loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True)
        self.optim = tf.keras.optimizers.SGD(0.0, 0.9, nesterov=True)
#         self.optim = tf.keras.optimizers.Adam(0.0,
#                                               beta_1=FLAGS.adam_beta_1,
#                                               beta_2=FLAGS.adam_beta_2,
#                                               epsilon=FLAGS.adam_epsilon)
#         if FLAGS.amp:
#             self.optim = tf.keras.mixed_precision.experimental.LossScaleOptimizer(
#                 self.optim, "dynamic")

        super(ClassificationRunner, self).__init__({'pyramidnet': self.model},
                                                   train_dataset,
                                                   valid_dataset=valid_dataset,
                                                   train_metrics=train_metrics,
                                                   valid_metrics=valid_metrics,
                                                   best_state='acc')

#     def begin_fit_callback(self, lr):
#         self.init_lr = lr
#         self.lr_scheduler = tf.keras.experimental.CosineDecay(self.init_lr, None)
#         self.optim.lr = self.init_lr

    def begin_epoch_callback(self, epoch_id, epochs):
#         if epoch_id<FLAGS.warmup:
#             self.optim.lr = epoch_id/FLAGS.warmup * self.init_lr
#         else:
#             self.lr_scheduler.decay_steps = epochs
#             self.optim.lr = self.lr_scheduler(epoch_id)
        
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

            if FLAGS.amp:
                scaled_loss = self.optim.get_scaled_loss(total_loss)
                scaled_grads = tape.gradient(scaled_loss,
                                             self.model.trainable_weights)
                grads = self.optim.get_unscaled_gradients(scaled_grads)
            else:
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
