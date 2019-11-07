import os
import time
import tensorflow as tf
from absl import flags, logging

FLAGS = flags.FLAGS
LOGGER = logging.get_absl_logger()


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
        self.train_num_batch = None
        self.valid_num_batch = None
        self.timer = time.time()
        self.best_state = best_state
        self.best_state_policy, self.best_record = self._state_policy_mapper(
            best_state)

    def add_metrics(self, name, tf_metrics, training):
        self.metrics[self._get_key(training)][name] = tf_metrics

    def append_message(self, msg):
        self.message = self.message + msg

    def add_scalar(self, key, value, epoch, training):
        with self.boards_writer[self._get_key(training)].as_default():
            tf.summary.scalar(key, value, step=epoch)

    def show_message(self, epoch, tensorboard=True):
        self.append_message('\nepoch: {}  '.format(epoch))
        time_cost = time.time() - self.timer
        results = self.get_result()
        tmp_msg = ''
        for key in self.keys:
            for k, v in results[key].items():
                tmp_msg = tmp_msg + '{}_{}: {:.4f}  '.format(key, k, v.numpy())
                if tensorboard:
                    with self.boards_writer[key].as_default():
                        tf.summary.scalar(self.metrics[key][k].name,
                                          v,
                                          step=epoch)
        if self.train_num_batch is None or self.valid_num_batch is None:
            tmp_msg = tmp_msg + 'samples/sec: unknown\n'
        else:
            tmp_msg = tmp_msg + 'samples/sec: {:.4f}\n'.format(
                (self.train_num_batch + self.valid_num_batch) * FLAGS.bs /
                time_cost)
        self.append_message(tmp_msg)
        LOGGER.info(self.message)

    def show_image(self, x, training, epoch):
        with self.boards_writer[self._get_key(training)].as_default():
            tf.summary.image('image', x, step=epoch, max_outputs=3)

    def update(self, data, training):
        """
        Args
            data (dict): key(str), value(list)
        """
        if type(data) == dict:
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
                self.metrics[key][k].reset_states()

        self.timer = time.time()
        self.num_data = 0
        self.message = ''

    def set_num_batch(self, train_num_batch, valid_num_batch):
        self.train_num_batch = train_num_batch
        self.valid_num_batch = valid_num_batch

    def _update_best_record(self, new_record):
        old_best_record = self.best_record
        self.best_record = self.best_state_policy(self.best_record, new_record)
        if self.best_record != old_best_record:
            tmp_msg = 'Best Performance at {}: {:.4f}'.format(
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
