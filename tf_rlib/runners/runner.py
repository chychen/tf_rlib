import os
import time
import numpy as np
from tqdm.auto import tqdm
from absl import flags, logging
import tensorflow as tf
from tensorflow.python.eager import profiler
from tf_rlib.runners import MetricsManager
from tf_rlib import utils

FLAGS = flags.FLAGS
LOGGER = logging.get_absl_logger()


class Runner:
    """ please make sure all the losses follow the distributed training mechanism:
    please see https://www.tensorflow.org/tutorials/distribute/custom_training
    """
    def __init__(self, train_dataset, valid_dataset=None, best_state=None):
        """
        Args
            models (dict): key(str), value(tf.keras.)
            metrics (dict): key(str), value(tf.keras.metrics.Metric)
        """
        utils.init_tf_rlib(show=True)
        self.strategy = tf.distribute.MirroredStrategy()
        LOGGER.info('Number of devices: {}'.format(
            self.strategy.num_replicas_in_sync))
        with self.strategy.scope():
            self.models, train_metrics, valid_metrics = self.init()
            self.train_dataset = self.strategy.experimental_distribute_dataset(
                train_dataset)
            self.valid_dataset = self.strategy.experimental_distribute_dataset(
                valid_dataset)

            # weights init in first call()
            for key, model in self.models.items():
                _ = model(
                    tf.keras.Input(
                        shape=train_dataset.element_spec[0].shape[1:],
                        dtype=tf.float32))
                LOGGER.info('{} model contains {} trainable variables.'.format(
                    key, model.num_params))
            self.epoch = 0
            self.step = 0
            self.save_path = FLAGS.save_path
            self.best_state = best_state
            self.matrics_manager = MetricsManager(best_state)
            if train_metrics is None or valid_metrics is None:
                raise ValueError(
                    'metrics are required, Note: please use tf.keras.metrics.MeanTensor to compute the training loss, which is more efficient by avoiding redundant tain loss computing.'
                )
            else:
                for k, v in train_metrics.items():
                    self.matrics_manager.add_metrics(k, v, training=True)
                for k, v in valid_metrics.items():
                    self.matrics_manager.add_metrics(k, v, training=False)

    def init(self):
        raise NotImplementedError

    def train_step(self, x, y):
        """ NOTE: drop_remainder=False in dataset will introduce bias in training 
        Args:
            x (batch): mini batch data
            y (batch): mini batch label
        Returns:
            losses (dict)
        """
        raise NotImplementedError

    @tf.function
    def _train_step(self, x, y):
        def train_fn(x, y):
            metrics = self.train_step(x, y)
            self.matrics_manager.update(metrics, training=True)

        self.strategy.experimental_run_v2(train_fn, args=(x, y))

    def validate_step(self, x, y):
        """
        Args:
            x (batch): mini batch data
            y (batch): mini batch label
        Returns:
            metrics (dict)
        """
        raise NotImplementedError

    @tf.function
    def _validate_step(self, x, y):
        def valid_fn(x, y):
            metrics = self.validate_step(x, y)
            self.matrics_manager.update(metrics, training=False)

        self.strategy.experimental_run_v2(valid_fn, args=(x, y))

    @tf.function
    def inference(self, dataset):
        raise NotImplementedError

    def begin_fit_callback(self, lr):
        pass

    def begin_epoch_callback(self, epoch_id, epochs, lr):
        pass

    def fit(self, epochs, lr):
        with self.strategy.scope():
            self.begin_fit_callback(lr)
            train_pbar = tqdm(desc='train', leave=False)
            valid_pbar = tqdm(desc='valid', leave=False)
            for e_idx in range(epochs):
                first_e_timer = time.time()
                self.begin_epoch_callback(self.epoch, epochs)
                self.epoch = self.epoch + 1
                # progress bars
                train_pbar.reset()
                valid_pbar.reset()
                self.matrics_manager.reset()
                # begin_epoch_callback
                # train one epoch
                for train_num_batch, (x_batch, y_batch) in enumerate(
                        self.train_dataset):
                    self.step = self.step + 1
                    if FLAGS.profile:
                        with profiler.Profiler(
                                os.path.join(FLAGS.log_path, 'profile')):
                            self._train_step(x_batch, y_batch)
                    else:
                        self._train_step(x_batch, y_batch)
                    train_pbar.update(1)
                self._log_data(x_batch, training=True)

                # validate one epoch
                if self.valid_dataset is not None:
                    for valid_num_batch, (x_batch, y_batch) in enumerate(
                            self.valid_dataset):
                        self._validate_step(x_batch, y_batch)
                        valid_pbar.update(1)
                    if self.matrics_manager.is_better_state():
                        self.save_best()
                        valid_pbar.set_postfix({
                            'best epoch':
                            self.epoch,
                            self.best_state:
                            self.matrics_manager.best_record
                        })
                    self._log_data(x_batch, training=False)

                if self.epoch == 0:
                    LOGGER.warn('time cost for first epoch: {} sec'.format(
                        time.time() - first_e_timer))
                if e_idx == 0:
                    train_num_batch = train_num_batch + 1
                    valid_num_batch = valid_num_batch + 1
                    self.matrics_manager.set_num_batch(train_num_batch,
                                                       valid_num_batch)
                    train_pbar.total = train_num_batch
                    valid_pbar.total = valid_num_batch

                # logging
                self.matrics_manager.show_message(self.epoch)

    def save(self, path):
        for key, model in self.models.items():
            model.save_weights(os.path.join(path, key))

    def load(self, path):
        for key, model in self.models.items():
            model.load_weights(os.path.join(path, key))

    def save_best(self):
        self.save(os.path.join(self.save_path, 'best'))

    def load_best(self):
        self.load(os.path.join(self.save_path, 'best'))

    def log_scalar(self, key, value, step, training):
        self.matrics_manager.add_scalar(key, value, step, training)

    @property
    def best_state_record(self):
        return self.matrics_manager.best_record

    def _get_size(self, dataset):
        num_batch = 0
        num_data = 0
        if dataset is not None:
            for data in dataset:
                num_batch += 1
                num_data += data.shape[0]
        return num_batch, num_data

    def _log_data(self, x_batch, training):
        if FLAGS.dim == 2:
            if self.strategy.num_replicas_in_sync == 1:
                x_batch_local = x_batch
            else:
                x_batch_local = x_batch.values[0]
            self.matrics_manager.show_image(x_batch_local,
                                            training=training,
                                            epoch=self.epoch)
