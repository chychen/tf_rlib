import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow_hub as hub
from tf_rlib.models.research import Predictor34 as Predictor, RandomNet34 as RandomNet
from tf_rlib.runners.base import runner
from tf_rlib.datasets import SVDBlurCifar10vsSVHN
from tf_rlib import losses, metrics
from tf_rlib import utils
from absl import flags
from absl import logging
import numpy as np

FLAGS = flags.FLAGS


class OODSvdRndCifar10vsSVHN(runner.Runner):
    """ [Novelty Detection Via Blurring, ICLR 2020](https://arxiv.org/abs/1911.11943)
    
    Performance: tnr@95tpr = 98.6~99.3 (96.9% on paper)
    Batch Size: 128
    Epochs: 10
    svd_remove: 28
    Dataset: SVDBlurCifar10vsSVHN
    Optimizer: AdamW + WeightDecay=0.0
    Scheduled LR: 1e-3 + CosineAnealing
    Trainable Parameters: 21,594,752
    Non-Trainable Parameters of 2 Random Network: 2 x 21,445,824
    """
    def __init__(self):
        utils.set_gpus('0')
        FLAGS.svd_remove = 28
        FLAGS.bs = 128
        FLAGS.dim = 2
        FLAGS.epochs = 10
        FLAGS.warmup = 0
        FLAGS.lr = 1e-3
        self.train_dataset, self.valid_dataset = SVDBlurCifar10vsSVHN(
        ).get_data()
        super(OODSvdRndCifar10vsSVHN,
              self).__init__(self.train_dataset,
                             valid_dataset=self.valid_dataset,
                             best_state='tnr@95tpr')
        # start training
        self.fit(FLAGS.epochs, FLAGS.lr)

    def init(self):
        valid_amount = 0
        for x_batch, _ in self.valid_dataset:
            valid_amount = valid_amount + x_batch.shape[0]

        input_shape = self.train_dataset.element_spec[0][0].shape[
            1:]  # [0][0] -> ((x, x_blur),y) -> x
        self.predictor = Predictor()
        self.randnet_1 = RandomNet()  # Note: fixed, not trainable
        self.randnet_2 = RandomNet()  # Note: fixed, not trainable
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
            loss1 = self.loss_object(g1, f1)
            loss2 = self.loss_object(g2, f2)
            loss = (loss1 + loss2) / 2.0
            # distributed-aware
            loss = tf.nn.compute_average_loss(loss, global_batch_size=FLAGS.bs)
            regularization_loss = tf.nn.scale_regularization_loss(
                tf.math.add_n(self.predictor.losses))
            total_loss = loss + regularization_loss

        trainable_w = self.predictor.trainable_weights
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
        f = self.predictor(x, training=False)
        g = self.randnet_1(x, training=False)
        loss = self.loss_object(g, f)
        # TODO distributed-aware
        #         loss = tf.nn.compute_average_loss(loss, global_batch_size=FLAGS.bs)
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
        return False
