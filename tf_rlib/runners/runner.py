import os
import time
import numpy as np
from tqdm.auto import tqdm
from absl import flags, logging
import tensorflow as tf
from tensorflow.python.eager import profiler
from tf_rlib.runners import MetricsManager

FLAGS = flags.FLAGS


class Runner:
    def __init__(self,
                 models,
                 train_dataset,
                 valid_dataset=None,
                 save_path=None,
                 train_metrics=None,
                 valid_metrics=None,
                 best_state=None):
        """
        Args
            models (dict): key(str), value(tf.keras.)
            metrics (dict): key(str), value(tf.keras.metrics.Metric)
        """
        self.models = models
        # weights init in first call()
        for key, model in self.models.items():
            _ = model(
                tf.keras.Input(shape=train_dataset.element_spec[0].shape[1:],
                               dtype=tf.float32))
            logging.info(self.model.num_params)
        self.epoch = 0
        self.step = 0
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.save_path = FLAGS.save_path
        self.train_dataset_size = self._get_size(train_dataset)
        self.valid_dataset_size = self._get_size(valid_dataset)
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

    @tf.function
    def train_step(self, x, y):
        """
        Args:
            x (batch): mini batch data
            y (batch): mini batch label
        Returns:
            losses (dict)
        """
        raise NotImplementedError

    @tf.function
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
    def inference(self, dataset):
        raise NotImplementedError

    def begin_fit_callback(self):
        pass

    def begin_epoch_callback(self, epoch_id, epochs, init_lr):
        pass

    def fit(self, epochs, lr=FLAGS.lr):
        self.begin_fit_callback(lr)
        self.train_pbar = tqdm(desc='train',
                               total=self.train_dataset_size,
                               leave=False)
        self.valid_pbar = tqdm(desc='valid',
                               total=self.valid_dataset_size,
                               leave=False)
        for _ in range(epochs):
            self.begin_epoch_callback(self.epoch, epochs)
            self.epoch = self.epoch + 1
            # progress bars
            self.train_pbar.reset(self.train_dataset_size)
            self.valid_pbar.reset(self.valid_dataset_size)
            self.matrics_manager.reset()
            # begin_epoch_callback
            # train one epoch
            for _, (x_batch, y_batch) in enumerate(self.train_dataset):
                self.step = self.step + 1
                if FLAGS.profile:
                    with profiler.Profiler(
                            os.path.join(FLAGS.log_path, 'profile')):
                        metrics = self.train_step(x_batch, y_batch)
                else:
                    metrics = self.train_step(x_batch, y_batch)
                self.matrics_manager.update(metrics, training=True)
                self.train_pbar.update(1)
            if FLAGS.dim == 2:
                self.matrics_manager.show_image(x_batch,
                                                training=True,
                                                epoch=self.epoch)
            # validate one epoch
            if self.valid_dataset is not None:
                for _, (x_batch, y_batch) in enumerate(self.valid_dataset):
                    metrics = self.validate_step(x_batch, y_batch)
                    self.matrics_manager.update(metrics, training=False)
                    self.valid_pbar.update(1)
                if self.matrics_manager.is_better_state():
                    self.save_best()
                    self.valid_pbar.set_postfix({
                        'best epoch':
                        self.epoch,
                        self.best_state:
                        self.matrics_manager.best_record
                    })
                if FLAGS.dim == 2:
                    self.matrics_manager.show_image(x_batch,
                                                    training=False,
                                                    epoch=self.epoch)
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

    def _get_size(self, dataset):
        num_elements = 0
        if dataset is not None:
            for _ in dataset:
                num_elements += 1
        return num_elements
