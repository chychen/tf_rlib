import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import os
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
from absl import flags, logging
from tqdm.auto import tqdm
from PIL import Image
from tf_rlib import datasets

FLAGS = flags.FLAGS
LOGGER = logging.get_absl_logger()


class Cifar10Semi(datasets.Dataset):
    """ use tfrecords could speed-up 5%~15% comparing to numpy. because tfrecords format only use 50% space than numpy.
    """
    def __init__(self, labels_persentage):
        super(Cifar10Semi, self).__init__()
        self.labels_persentage = labels_persentage
        self.dtype = np.float16 if FLAGS.amp else np.float32
        save_path = '/ws_data/tmp/cifar10_labels{}'.format(labels_persentage)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        if FLAGS.amp:
            self.train_file = os.path.join(save_path, 'train16.tfrecords')
            self.valid_file = os.path.join(save_path, 'valid16.tfrecords')
        else:
            self.train_file = os.path.join(save_path, 'train32.tfrecords')
            self.valid_file = os.path.join(save_path, 'valid32.tfrecords')

        if not os.path.exists(self.train_file):
            train_np, valid_np = self._get_np_dsets()
            with tf.io.TFRecordWriter(self.train_file) as writer:
                for image, label in tqdm(zip(*train_np), total=50000):
                    image_string = image.tobytes()
                    tf_example = self.image_example(image_string, label)
                    writer.write(tf_example.SerializeToString())

            with tf.io.TFRecordWriter(self.valid_file) as writer:
                for image, label in tqdm(zip(*valid_np), total=10000):
                    image_string = image.tobytes()
                    tf_example = self.image_example(image_string, label)
                    writer.write(tf_example.SerializeToString())

    # Create a dictionary with features that may be relevant.
    def image_example(self, image_string, label):
        def _bytes_feature(value):
            """Returns a bytes_list from a string / byte."""
            if isinstance(value, type(tf.constant(0))):
                value = value.numpy(
                )  # BytesList won't unpack a string from an EagerTensor.
            return tf.train.Feature(bytes_list=tf.train.BytesList(
                value=[value]))

        def _int64_feature(value):
            """Returns an int64_list from a bool / enum / int / uint."""
            return tf.train.Feature(int64_list=tf.train.Int64List(
                value=[value]))

        feature = {
            'label': _int64_feature(label),
            'image_raw': _bytes_feature(image_string)
        }
        return tf.train.Example(features=tf.train.Features(feature=feature))

    def get_data(self):
        return self._get_tf_dsets()

    def _get_np_dsets(self):
        train_data, valid_data = tf.keras.datasets.cifar10.load_data()

        train_data_x = train_data[0].astype(np.float32)
        valid_data_x = valid_data[0].astype(np.float32)
        train_data_y = train_data[1]
        valid_data_y = valid_data[1]
        labels_amount = int(len(train_data_x) * self.labels_persentage / 10.)
        new_x = []
        new_y = []
        for i in range(10):
            new_x.append(train_data_x[train_data_y[:, 0] == i][:labels_amount])
            new_y.append(train_data_y[train_data_y[:, 0] == i][:labels_amount])
        train_data_x = np.concatenate(new_x, axis=0)
        train_data_y = np.concatenate(new_y, axis=0)
        mean = train_data_x.mean(axis=(0, 1, 2))
        stddev = train_data_x.std(axis=(0, 1, 2))
        train_data_x = (train_data_x - mean) / stddev
        valid_data_x = (valid_data_x - mean) / stddev
        train_data_x = train_data_x.astype(self.dtype)
        valid_data_x = valid_data_x.astype(self.dtype)
        return (train_data_x, train_data_y), (valid_data_x, valid_data_y)

    def _get_tf_dsets(self):
        # Create a dictionary describing the features.
        image_feature_description = {
            'label': tf.io.FixedLenFeature([], tf.int64),
            'image_raw': tf.io.FixedLenFeature([], tf.string),
        }

        @tf.function
        def augmentation(x, y, pad=4):
            x = tf.image.resize_with_crop_or_pad(x, 32 + pad * 2, 32 + pad * 2)
            x = tf.image.random_crop(x, [32, 32, 3])
            x = tf.image.random_flip_left_right(x)
            return x, y

        @tf.function
        def parse(example_proto):
            example = tf.io.parse_single_example(example_proto,
                                                 image_feature_description)
            x = tf.io.decode_raw(example['image_raw'], self.dtype)
            x = tf.reshape(x, [32, 32, 3])
            return x, example['label']

        train_dataset = tf.data.TFRecordDataset(self.train_file)
        train_dataset = train_dataset.map(
            parse,
            num_parallel_calls=tf.data.experimental.AUTOTUNE).cache().map(
                augmentation,
                num_parallel_calls=tf.data.experimental.AUTOTUNE).shuffle(
                    50000).batch(FLAGS.bs, drop_remainder=True).prefetch(
                        buffer_size=tf.data.experimental.AUTOTUNE)

        valid_dataset = tf.data.TFRecordDataset(self.valid_file)
        valid_dataset = valid_dataset.map(
            parse,
            num_parallel_calls=tf.data.experimental.AUTOTUNE).cache().batch(
                FLAGS.bs, drop_remainder=False).prefetch(
                    buffer_size=tf.data.experimental.AUTOTUNE)
        return [train_dataset, valid_dataset]


class Cifar10Rotate(datasets.Dataset):
    """ use tfrecords could speed-up 5%~15% comparing to numpy. because tfrecords format only use 50% space than numpy.
    """
    def __init__(self):
        super(Cifar10Rotate, self).__init__()
        self.dtype = np.float16 if FLAGS.amp else np.float32
        save_path = '/ws_data/tmp/cifar10'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        if FLAGS.amp:
            self.train_file = os.path.join(save_path, 'train16.tfrecords')
            self.valid_file = os.path.join(save_path, 'valid16.tfrecords')
        else:
            self.train_file = os.path.join(save_path, 'train32.tfrecords')
            self.valid_file = os.path.join(save_path, 'valid32.tfrecords')

        if not os.path.exists(self.train_file):
            train_np, valid_np = self._get_np_dsets()
            with tf.io.TFRecordWriter(self.train_file) as writer:
                for image, label in tqdm(zip(*train_np), total=50000):
                    image_string = image.tobytes()
                    tf_example = self.image_example(image_string, label)
                    writer.write(tf_example.SerializeToString())

            with tf.io.TFRecordWriter(self.valid_file) as writer:
                for image, label in tqdm(zip(*valid_np), total=10000):
                    image_string = image.tobytes()
                    tf_example = self.image_example(image_string, label)
                    writer.write(tf_example.SerializeToString())

    # Create a dictionary with features that may be relevant.
    def image_example(self, image_string, label):
        def _bytes_feature(value):
            """Returns a bytes_list from a string / byte."""
            if isinstance(value, type(tf.constant(0))):
                value = value.numpy(
                )  # BytesList won't unpack a string from an EagerTensor.
            return tf.train.Feature(bytes_list=tf.train.BytesList(
                value=[value]))

        def _int64_feature(value):
            """Returns an int64_list from a bool / enum / int / uint."""
            return tf.train.Feature(int64_list=tf.train.Int64List(
                value=[value]))

        feature = {
            'label': _int64_feature(label),
            'image_raw': _bytes_feature(image_string)
        }
        return tf.train.Example(features=tf.train.Features(feature=feature))

    def get_data(self):
        return self._get_tf_dsets()

    def _get_np_dsets(self):
        train_data, valid_data = tf.keras.datasets.cifar10.load_data()
        train_data_x = train_data[0].astype(np.float32)
        valid_data_x = valid_data[0].astype(np.float32)
        mean = train_data_x.mean(axis=(0, 1, 2))
        stddev = train_data_x.std(axis=(0, 1, 2))
        train_data_x = (train_data_x - mean) / stddev
        valid_data_x = (valid_data_x - mean) / stddev
        train_data_x = train_data_x.astype(self.dtype)
        valid_data_x = valid_data_x.astype(self.dtype)
        train_data_y = train_data[1]
        valid_data_y = valid_data[1]
        return (train_data_x, train_data_y), (valid_data_x, valid_data_y)

    def _get_tf_dsets(self):
        # Create a dictionary describing the features.
        image_feature_description = {
            'label': tf.io.FixedLenFeature([], tf.int64),
            'image_raw': tf.io.FixedLenFeature([], tf.string),
        }

        @tf.function
        def rotate(x, y, pad=4):
            rot_y = tf.random.uniform(shape=[],
                                      minval=0,
                                      maxval=3,
                                      dtype=tf.int32)
            x = tf.image.rot90(x, k=rot_y)
            return x, rot_y

        @tf.function
        def parse(example_proto):
            example = tf.io.parse_single_example(example_proto,
                                                 image_feature_description)
            x = tf.io.decode_raw(example['image_raw'], self.dtype)
            x = tf.reshape(x, [32, 32, 3])
            return x, example['label']

        train_dataset = tf.data.TFRecordDataset(self.train_file)
        train_dataset = train_dataset.map(
            parse,
            num_parallel_calls=tf.data.experimental.AUTOTUNE).cache().map(
                rotate,
                num_parallel_calls=tf.data.experimental.AUTOTUNE).shuffle(
                    50000).batch(FLAGS.bs, drop_remainder=True).prefetch(
                        buffer_size=tf.data.experimental.AUTOTUNE)

        valid_dataset = tf.data.TFRecordDataset(self.valid_file)
        valid_dataset = valid_dataset.map(
            parse, num_parallel_calls=tf.data.experimental.AUTOTUNE).map(
                rotate, num_parallel_calls=tf.data.experimental.AUTOTUNE
            ).cache().batch(  # cache rotate
                FLAGS.bs, drop_remainder=False).prefetch(
                    buffer_size=tf.data.experimental.AUTOTUNE)
        return [train_dataset, valid_dataset]
