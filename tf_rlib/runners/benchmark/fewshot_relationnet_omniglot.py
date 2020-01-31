""" Features
- [v] PyramidNet
- [v] Warmup LR
- [v] CosineAnnealing LR
- [ ] Mixup
- [ ] AdamW
- [ ] Fixup Init
- [ ] Lookahead
- [v] WeightDecay
"""
import tensorflow as tf
import tensorflow_addons as tfa
from tf_rlib.models import RN_Omniglot
from tf_rlib.runners.base import runner
from tf_rlib.datasets import Omniglot
from tf_rlib.losses import MSELoss
from tf_rlib import utils
from absl import flags
from absl import logging
import numpy as np

FLAGS = flags.FLAGS


class FewShotRelationNetOmniglot(runner.Runner):
    """
    train_episodes_per_epoch = 1200 classes
    valid_episodes_per_epoch = 423 classes
        
    Dataset: Omniglot
    Model: Relation Network (paper)
    decay_per_episodes: 1e5
    Scheduled LR: 1e-3 (0-1e5), 1e-4 (1e5-2e5), 1e-5 (2e5-?)
    Optimizer: Adam
    Accuracy%: 
        5-way, 1-shot: 98.5%
        5-way, 5-shot: 99.6%
    Parameters: 11,173,962
    """
    def __init__(self):
        # change visible gpu immediately before dataset api seeing them
        utils.set_gpus('0')
        FLAGS.dim = 2
        FLAGS.c_way = 5
        FLAGS.k_shot = 5
        FLAGS.bs = 1
        FLAGS.log_level = 'WARN'
        self.dset = Omniglot()
        train_dataset, valid_dataset = self.dset.get_data()
        super(FewShotRelationNetOmniglot,
              self).__init__(train_dataset,
                             valid_dataset=valid_dataset,
                             best_state='acc')
        self.fit(300, lr=1e-3)

    def init(self):
        self.model = RN_Omniglot()
        train_metrics = {
            'loss': tf.keras.metrics.Mean('loss'),
            'acc': tf.keras.metrics.CategoricalAccuracy('acc')
        }
        valid_metrics = {
            'loss': tf.keras.metrics.Mean('loss'),
            'acc': tf.keras.metrics.CategoricalAccuracy('acc')
        }
        self.loss_object = MSELoss()
        self.optim = tf.keras.optimizers.Adam(lr=FLAGS.lr,
                                              beta_1=FLAGS.adam_beta_1,
                                              beta_2=FLAGS.adam_beta_2,
                                              epsilon=FLAGS.adam_epsilon)
        return {'relation_network': self.model}, train_metrics, valid_metrics

    def begin_fit_callback(self, lr):
        self.init_lr = lr
        decay_per_episodes = 100000
        decay_per_epochs = decay_per_episodes // self.dset.n_train_episode // FLAGS.bs
        self.learning_rate_fn = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
            [decay_per_epochs, 2 * decay_per_epochs], [1e-3, 1e-4, 1e-5])

    def begin_epoch_callback(self, epoch_id, epochs):
        self.optim.learning_rate = self.learning_rate_fn(epoch_id)
        self.log_scalar('lr', self.optim.learning_rate, training=True)

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
        loss = tf.nn.compute_average_loss(
            loss, global_batch_size=FLAGS.bs)  # distributed-aware
        probs = tf.nn.softmax(logits)
        return {'loss': [loss], 'acc': [y, probs]}

    @property
    def required_flags(self):
        pass
