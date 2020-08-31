import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.utils import to_categorical
from absl import flags, logging
from tqdm.auto import tqdm
from tf_rlib import datasets
from glob import glob
from datetime import datetime

FLAGS = flags.FLAGS
LOGGER = logging.get_absl_logger()


class DopplerWind(datasets.Dataset):
    def __init__(self, gridsize=32):  #TODO
        super(DopplerWind, self).__init__()
        self.gridsize = gridsize
        self.root = f'/ws_data/CWB/doppler_wind/tfrec_{gridsize}'
        mean_x = np.load('/ws_data/CWB/doppler_wind/mean_x.npy')  # (16,)
        mean_x = np.concatenate([mean_x, np.zeros(
            (14, ))])  # (16+14,), 14: mask
        std_x = np.load('/ws_data/CWB/doppler_wind/std_x.npy')  # (16,)
        std_x = np.concatenate([std_x, np.ones((14, ))])  # (16+14,), 14: mask
        self.tally = {
            'mean_x': mean_x,
            'std_x': std_x,
            'mean_y': np.load('/ws_data/CWB/doppler_wind/mean_y.npy'),
            'std_y': np.load('/ws_data/CWB/doppler_wind/std_y.npy')
        }

    def get_data(self):
        return self._get_tf_dsets()

    def get_y_denorm_fn(self):
        return lambda x: x * self.tally['std_y'] + self.tally['mean_y']

    def _get_tf_dsets(self):
        # Create a dictionary describing the features.
        image_feature_description = {
            'y': tf.io.FixedLenFeature([], tf.string),
            'x': tf.io.FixedLenFeature([], tf.string),
        }

        @tf.function
        def parse(example_proto):
            example = tf.io.parse_single_example(example_proto,
                                                 image_feature_description)
            x = tf.io.decode_raw(example['x'], np.float32)
            x = tf.reshape(x, [self.gridsize, self.gridsize, 16 + 14
                               ])  # NOTE: FIXED # 14: mask
            y = tf.io.decode_raw(example['y'], np.float32)
            y = tf.reshape(y,
                           [self.gridsize, self.gridsize, 28])  # NOTE: FIXED
            # Norm
            x = (x - self.tally['mean_x']) / self.tally['std_x']
            y = (y - self.tally['mean_y']) / self.tally['std_y']

            if FLAGS.amp:
                x = tf.cast(x, tf.float16)
                y = tf.cast(y, tf.float16)
            return x, y

        # TODO Augmentation and Normalization, cache on training?

        def get_tfrec_dset(is_train: bool):
            folder = 'train' if is_train else 'test'
            files = tf.io.matching_files(
                os.path.join(self.root, f'{folder}/*.tfrec'))
            shards = tf.data.Dataset.from_tensor_slices(files)
            shards = shards.shuffle(files.numpy().shape[0])
            shards = shards.interleave(
                lambda x: tf.data.TFRecordDataset(x),
                num_parallel_calls=tf.data.experimental.AUTOTUNE,
                block_length=1,
                #             cycle_length=4, # set by num_parallel_calls
                deterministic=True)
            shards = shards.map(
                parse, num_parallel_calls=tf.data.experimental.AUTOTUNE)
            return shards

        train_dataset = get_tfrec_dset(is_train=True)
        train_dataset = train_dataset.shuffle(10000).batch(
            FLAGS.bs, drop_remainder=True).prefetch(
                buffer_size=tf.data.experimental.AUTOTUNE)
        if FLAGS.steps_per_epoch is not None:
            train_dataset = train_dataset.take(FLAGS.steps_per_epoch)

        valid_dataset = get_tfrec_dset(is_train=False)
        valid_dataset = valid_dataset.cache().batch(
            FLAGS.bs, drop_remainder=False).prefetch(
                buffer_size=tf.data.experimental.AUTOTUNE)

        return [train_dataset, valid_dataset]
