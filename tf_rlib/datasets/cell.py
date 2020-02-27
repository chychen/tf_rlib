import tensorflow as tf
from tensorflow.keras.utils import to_categorical
import numpy as np
from tf_rlib import datasets
from absl import flags, logging

FLAGS = flags.FLAGS
LOGGER = logging.get_absl_logger()


class CellSegmentation(datasets.Dataset):
    '''
    Mount point: Warren_datasets/SegBenchmark/medical/cell/
    '''
    def __init__(self, path='/mount/data/SegBenchmark/medical/cell/'):
        self.path = path
        super(CellSegmentation, self).__init__()

    def get_data(self):
        return self._get_dsets()

    def _get_dsets(self):
        X = np.load(self.path + 'train/X.npy')
        Y = np.load(self.path + 'train/Y.npy')[..., None]
        Y = to_categorical(Y, 2)
        train_data_x, valid_data_x = X[:int(len(X) * .8)], X[int(len(X) * .8):]
        train_data_y, valid_data_y = Y[:int(len(X) * .8)], Y[int(len(X) * .8):]

        mean = train_data_x.mean(axis=(0, 1, 2))
        stddev = train_data_x.std(axis=(0, 1, 2))
        train_data_x = (train_data_x - mean) / stddev
        valid_data_x = (valid_data_x - mean) / stddev
        logging.info('mean:{}, std:{}'.format(mean, stddev))
        logging.info('data size:{}, label size"{}'.format(
            train_data_x.shape, train_data_y.shape))

        @tf.function
        def augmentation(x, y, pad=4):
            flip = tf.random.uniform([1], 0, 1)[0]
            # random flip
            if flip > .67:
                x = tf.image.flip_up_down(x)
                y = tf.image.flip_up_down(y)
            elif flip > .33:
                x = tf.image.flip_left_right(x)
                y = tf.image.flip_left_right(y)
            else:
                pass

            return x, y

        train_dataset = tf.data.Dataset.from_tensor_slices(
            (train_data_x, train_data_y))
        valid_dataset = tf.data.Dataset.from_tensor_slices(
            (valid_data_x, valid_data_y))
        train_dataset = train_dataset.map(
            augmentation,
            num_parallel_calls=tf.data.experimental.AUTOTUNE).cache().shuffle(
                300).batch(FLAGS.bs, drop_remainder=True).prefetch(
                    buffer_size=tf.data.experimental.AUTOTUNE)
        valid_dataset = valid_dataset.cache().batch(
            FLAGS.bs, drop_remainder=False).prefetch(
                buffer_size=tf.data.experimental.AUTOTUNE)
        return [train_dataset, valid_dataset]
