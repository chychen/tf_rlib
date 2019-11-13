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
from tf_rlib.models import PyramidNet
from tf_rlib.runners import runner
from absl import flags
from absl import logging
import numpy as np

FLAGS = flags.FLAGS


class ClassificationRunner(runner.Runner):
    def __init__(self, train_dataset, valid_dataset=None):
        super(ClassificationRunner, self).__init__(train_dataset,
                                                   valid_dataset=valid_dataset,
                                                   best_state='acc')
        self.test_metrics = {
            'loss': tf.keras.metrics.Mean('loss'),# TODO
            'acc': tf.keras.metrics.SparseCategoricalAccuracy('acc')# TODO
        }

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
            total_loss = loss + regularization_loss
            total_loss = loss

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
    
    def _TTA(self, x, size):
        # augment
        pad_range = [4, 4]
        paddings = tf.constant([[0, 0], pad_range, pad_range, [0, 0]])
        x_pad = tf.pad(x, paddings, mode='CONSTANT')
        x_zi = x_pad[:, :size[0], :size[1]]
        x_zo = x_pad[:, -size[0]:, -size[1]:]
        # predict and average
        logits = self.model(x, training=False)
        logits_zi = self.model(x_zi, training=False)
        logits_zo = self.model(x_zo, training=False)
        return (logits+logits_zi+logits_zo)/3
    
    @tf.function
    def inference(self, x):
        """
        With zoom-in zoom-out TTA implementation
        """
        bs = FLAGS.bs
        iters = int(len(x)/bs)
        remain = len(x)%bs
        results = []
        for i in range(iters):
            data = x[i*bs:(i+1)*bs]
            size = data.shape[1:3]
            result = self._TTA(data, size)
            results.append(result)
        if remain:
            data = x[-remain:]
            size = data.shape[1:3]
            result = self._TTA(data, size)
            results.append(result)
        return tf.concat(results, axis=0)

    def evaluate_step(self, x, y):
        logits = self.inference(x)
        loss = self.loss_object(y, logits)
        loss = tf.nn.compute_average_loss(loss, global_batch_size=FLAGS.bs)
        probs = tf.nn.softmax(logits)
        self.test_metrics['loss'].update_state(loss) # TODO
        self.test_metrics['acc'].update_state(y, probs) # TODO
