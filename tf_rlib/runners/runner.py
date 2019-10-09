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
                 metrics=None,
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
        self.matrics_manager = MetricsManager(best_state)
        if metrics is None:
            raise ValueError(
                'metrics is required, please use tf.keras.metrics.MeanTensor for training loss.'
            )
        else:
            for k, v in metrics.items():
                self.matrics_manager.add_metrics(k, v)

    def fit(self, epochs):
        for _ in range(epochs):
            self.epoch = self.epoch + 1
            self.matrics_manager.reset()
            self.matrics_manager.append_message('epoch: {}\n'.format(self.epoch))
            # train one epoch
            with tqdm(total=self.train_dataset_size, leave=False) as pbar:
                for step, (x_batch, y_batch) in enumerate(self.train_dataset):
                    loss = self.train_step(x_batch, y_batch)
                    pbar.update(1)
                    self.matrics_manager.update(loss)
            # validate one epoch
            if self.valid_dataset is not None:
                self.evaluate(self.valid_dataset, self.valid_dataset_size)
            # logging
            self.matrics_manager.show_message()

    def evaluate(self, dataset, size=None, auto_save=True):
        if size is None:
            size = self._get_size(dataset)
        with tqdm(total=size, leave=False) as pbar:
            for step, (x_batch, y_batch) in enumerate(dataset):
                metrics = self.validate_step(x_batch, y_batch)
                self.matrics_manager.update(metrics)
                pbar.update(1)
            if self.matrics_manager.is_better_state() and auto_save:
                self.save_best()

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

    def validate_step(self, x, y):
        """
        Args:
            x (batch): mini batch data
            y (batch): mini batch label
        Returns:
            metrics (dict)
        """
        raise NotImplementedError

    def inference(self, dataset):
        raise NotImplementedError


class MetricsManager:
    def __init__(self, best_state):
        self.message = ''
        self.data = {}
        self.num_data = 0
        self.timer = time.time()
        self.best_state = best_state
        self.best_state_policy, self.best_record = self._state_policy_mapper(
            best_state)

    def add_metrics(self, name, tf_metrics):
        self.data[name] = tf_metrics

    def append_message(self, msg):
        self.message = self.message + msg

    def show_message(self):
        time_cost = time.time() - self.timer
        dict_ = self.get_result()
        tmp_msg = ''
        for k, v in dict_.items():
            tmp_msg = tmp_msg + '{}: {:.4f}  '.format(k, v)
        tmp_msg = tmp_msg + 'samples/sec: {:.4f}\n'.format(
            self.num_data * FLAGS.bs / time_cost)
        self.append_message(tmp_msg)
        logging.info(self.message)

    def update(self, data):
        """
        Args
            data (dict): key(str), value(list)
        """
        if type(data) == dict:
            self.num_data = self.num_data + 1
            for k, v in data.items():
                self.data[k].update_state(*v)
        else:
            raise ValueError

    def get_result(self):
        ret_dict = {}
        for k, v in self.data.items():
            ret_dict[k] = self.data[k].result().numpy()
        return ret_dict

    def is_better_state(self):
        results = self.get_result()
        if self.best_state not in results:
            raise ValueError(
                'You should pass metrics({}) as one of the return in validate_step()'
                .format(self.best_state))
        return self._update_best_record(results[self.best_state])

    def reset(self):
        for k, v in self.data.items():
            if k not in self.data:
                self.data[k].reset_states()
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

    def _state_policy_mapper(self, prefix_state):
        state = prefix_state.split('_')[-1]
        if state == 'acc' or state == 'precision' or state == 'recall' or state == 'f1':
            return max, float('-inf')
        if state == 'mae' or state == 'mse' or state == 'l1' or state == 'l2':
            return min, float('inf')
        else:
            raise ValueError
