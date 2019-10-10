import os
import time
import numpy as np
from tqdm.auto import tqdm
from absl import flags, logging
import tensorflow as tf

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
        self.epoch = 0
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

    def fit(self, epochs):
        self.train_pbar = tqdm(desc='train',
                               total=self.train_dataset_size,
                               leave=False)
        self.valid_pbar = tqdm(desc='valid',
                               total=self.valid_dataset_size,
                               leave=False)
        for _ in range(epochs):
            self.epoch = self.epoch + 1
            self.train_pbar.reset(self.train_dataset_size)
            self.valid_pbar.reset(self.valid_dataset_size)
            self.matrics_manager.reset()
            # train one epoch
            for step, (x_batch, y_batch) in enumerate(self.train_dataset):
                metrics = self.train_step(x_batch, y_batch)
                self.matrics_manager.update(metrics, training=True)
                self.train_pbar.update(1)
            # validate one epoch
            if self.valid_dataset is not None:
                for step, (x_batch, y_batch) in enumerate(self.valid_dataset):
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

    def _get_size(self, dataset):
        num_elements = 0
        if dataset is not None:
            for _ in dataset:
                num_elements += 1
        return num_elements

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


class MetricsManager:
    KEY_TRAIN = 'train'
    KEY_VALID = 'valid'

    def __init__(self, best_state):
        self.message = ''
        self.keys = [MetricsManager.KEY_TRAIN, MetricsManager.KEY_VALID]
        self.metrics = {self.keys[0]: dict(), self.keys[1]: dict()}
        train_log_path = os.path.join(FLAGS.log_path, self.keys[0])
        valid_log_path = os.path.join(FLAGS.log_path, self.keys[1])
        self.boards_writer = {
            self.keys[0]: tf.summary.create_file_writer(train_log_path),
            self.keys[1]: tf.summary.create_file_writer(valid_log_path)
        }
        self.num_data = 0
        self.timer = time.time()
        self.best_state = best_state
        self.best_state_policy, self.best_record = self._state_policy_mapper(
            best_state)

    def add_metrics(self, name, tf_metrics, training):
        self.metrics[self._get_key(training)][name] = tf_metrics

    def append_message(self, msg):
        self.message = self.message + msg

    def show_message(self, epoch, tensorboard=True):
        self.append_message('epoch: {}  '.format(epoch))
        time_cost = time.time() - self.timer
        results = self.get_result()
        tmp_msg = ''
        for key in self.keys:
            for k, v in results[key].items():
                tmp_msg = tmp_msg + '{}: {:.4f}  '.format(k, v.numpy())
                if tensorboard:
                    with self.boards_writer[key].as_default():
                        tf.summary.scalar(self.metrics[key][k].name,
                                          v,
                                          step=epoch)
        tmp_msg = tmp_msg + 'samples/sec: {:.4f}\n'.format(
            self.num_data * FLAGS.bs / time_cost)
        self.append_message(tmp_msg)
        logging.info(self.message)

    def update(self, data, training):
        """
        Args
            data (dict): key(str), value(list)
        """
        if type(data) == dict:
            self.num_data = self.num_data + 1
            for k, v in data.items():
                self.metrics[self._get_key(training)][k].update_state(*v)
        else:
            raise ValueError

    def get_result(self):
        ret_dict = {}
        for key in self.keys:
            ret_dict[key] = dict()
            for k, v in self.metrics[key].items():
                ret_dict[key][k] = self.metrics[key][k].result()
        return ret_dict

    def is_better_state(self):
        results = self.get_result()
        valid_results = results[MetricsManager.KEY_VALID]
        if self.best_state not in valid_results:
            raise ValueError(
                'You should pass metrics({}) as one of the return in validate_step()'
                .format(self.best_state))
        return self._update_best_record(valid_results[self.best_state].numpy())

    def reset(self):
        for key in self.keys:
            for k, v in self.metrics[key].items():
                if k not in self.metrics[key]:
                    self.metrics[key][k].reset_states()
        self.timer = time.time()
        self.num_data = 0
        self.message = ''

    def _update_best_record(self, new_record):
        old_best_record = self.best_record
        self.best_record = self.best_state_policy(self.best_record, new_record)
        if self.best_record != old_best_record:
            tmp_msg = 'Best Performance at {}: {:.4f}\n'.format(
                self.best_state, self.best_record)
            self.append_message(tmp_msg)
            return True
        else:
            return False

    def _state_policy_mapper(self, state):
        if state == 'acc' or state == 'precision' or state == 'recall' or state == 'f1':
            return max, float('-inf')
        if state == 'mae' or state == 'mse' or state == 'l1' or state == 'l2':
            return min, float('inf')
        else:
            raise ValueError

    def _get_key(self, training):
        key = MetricsManager.KEY_TRAIN if training else MetricsManager.KEY_VALID
        return key
