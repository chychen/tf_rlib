import os
import time
import numpy as np
import tensorflow as tf
from absl import flags, logging

FLAGS = flags.FLAGS
LOGGER = logging.get_absl_logger()


class MetricsManager:
    KEY_TRAIN = 'train'
    KEY_VALID = 'valid'
    KEY_TEST = 'test'

    TAG_METRICS = 'metrics/'
    TAG_HPARAMS = 'hyper-perameters/'

    def __init__(self, best_state):
        self.message = ''
        self.keys = [
            MetricsManager.KEY_TRAIN, MetricsManager.KEY_VALID,
            MetricsManager.KEY_TEST
        ]
        self.metrics = {
            self.keys[0]: dict(),
            self.keys[1]: dict(),
            self.keys[2]: dict()
        }
        train_log_path = os.path.join(FLAGS.log_path, self.keys[0])
        valid_log_path = os.path.join(FLAGS.log_path, self.keys[1])
        self.boards_writer = {
            self.keys[0]: tf.summary.create_file_writer(train_log_path),
            self.keys[1]: tf.summary.create_file_writer(valid_log_path)
        }
        markdown_str = '| key | value |\n|:-:|:-:|\n'
        for k, v in FLAGS.flag_values_dict().items():
            markdown_str += '| {} | {} |\n'.format(k, v)
        with self.boards_writer[MetricsManager.KEY_VALID].as_default():
            tf.summary.text('FLAGS',
                            data=tf.convert_to_tensor(markdown_str),
                            step=0)
        self.train_num_batch = None
        self.valid_num_batch = None
        self.timer = time.time()
        self.best_state = best_state
        self.best_state_policy, self.best_record = self._state_policy_mapper(
            best_state)

    def add_metrics(self, name, tf_metrics, key):
        self.metrics[key][name] = tf_metrics

    def append_message(self, msg):
        self.message = self.message + msg

    def add_scalar(self, name, value, epoch, key, tag=''):
        with self.boards_writer[key].as_default():
            tf.summary.scalar(tag + name, value, step=epoch)

    def show_message(self, epoch, tensorboard=True):
        self.append_message('\nepoch: {}  '.format(epoch))
        time_cost = time.time() - self.timer
        img_p_sec = 0.0
        results = self.get_result(
            keys=[MetricsManager.KEY_TRAIN, MetricsManager.KEY_VALID])
        tmp_msg = ''
        for key in [MetricsManager.KEY_TRAIN, MetricsManager.KEY_VALID]:
            for k, v in results[key].items():
                tmp_msg = tmp_msg + '{}_{}: {:.4f}  '.format(key, k, v.numpy())
                if tensorboard:
                    with self.boards_writer[key].as_default():
                        tf.summary.scalar(MetricsManager.TAG_METRICS +
                                          self.metrics[key][k].name,
                                          v,
                                          step=epoch)
        if self.train_num_batch is None or self.valid_num_batch is None:
            tmp_msg = tmp_msg + 'samples/sec: unknown\n'
        else:
            img_p_sec = (self.train_num_batch +
                         self.valid_num_batch) * FLAGS.bs / time_cost
            tmp_msg = tmp_msg + 'samples/sec: {:.4f}\n'.format(img_p_sec)
        self.append_message(tmp_msg)
        LOGGER.info(self.message)
        if tensorboard:
            self.add_scalar(MetricsManager.TAG_METRICS + 'best_record',
                            self.best_record, epoch, MetricsManager.KEY_VALID)
            self.add_scalar(MetricsManager.TAG_HPARAMS + 'img_p_sec',
                            img_p_sec, epoch, MetricsManager.KEY_VALID)

    def show_image(self, x, key, epoch, name='image'):
        with self.boards_writer[key].as_default():
            for i, v in enumerate(x):
                tf.summary.image(str(i) + '/' + name,
                                 v[None],
                                 step=epoch,
                                 max_outputs=1)

    def update(self, data, key):
        """
        Args
            data (dict): key(str), value(list)
        """
        if type(data) == dict:
            for k, v in data.items():
                self.metrics[key][k].update_state(*v)
        else:
            raise ValueError

    def get_result(self, keys=None):
        ret_dict = {}
        for key in keys:
            ret_dict[key] = dict()
            for k, v in self.metrics[key].items():
                ret_dict[key][k] = self.metrics[key][k].result()
        return ret_dict

    def is_better_state(self):
        results = self.get_result(
            keys=[MetricsManager.KEY_TRAIN, MetricsManager.KEY_VALID])
        valid_results = results[MetricsManager.KEY_VALID]
        if self.best_state not in valid_results:
            raise ValueError(
                'You should pass metrics({}) as one of the return in validate_step()'
                .format(self.best_state))
        return self._update_best_record(valid_results[self.best_state].numpy())

    def reset_metrics(self, key):
        for k, v in self.metrics[key].items():
            self.metrics[key][k].reset_states()

    def reset(self):
        for key in self.keys:
            self.reset_metrics(key)

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
        if state == 'acc' or state == 'precision' or state == 'recall' or state == 'f1' or 'auc' in state:
            return max, float('-inf')
        if state == 'mae' or state == 'mse' or state == 'l1' or state == 'l2' or state == 'loss':
            return min, float('inf')
        if state == 'dice_coef':
            return max, float('-inf')
        else:
            raise ValueError

    def register_hparams(self):
        from tensorboard.plugins.hparams import api as hp
        hparams = {}
        for k, v in FLAGS.flag_values_dict().items():
            if v is not None:
                hparams[k] = v
        with tf.summary.create_file_writer(FLAGS.log_path).as_default():
            hp.hparams(hparams)  # record the values used in this trial
