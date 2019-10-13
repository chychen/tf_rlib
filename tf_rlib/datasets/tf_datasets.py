import tensorflow as tf
import numpy as np
from absl import flags, logging

FLAGS = flags.FLAGS


def get_cifar10():
    train_data, valid_data = tf.keras.datasets.cifar10.load_data()
    train_data_x = train_data[0].astype(np.float32)
    valid_data_x = valid_data[0].astype(np.float32)
    mean = train_data_x.mean(axis=(0,1,2))
    stddev = train_data_x.std(axis=(0,1,2))
    train_data_x = (train_data_x - mean) / stddev
    valid_data_x = (valid_data_x - mean) / stddev
    train_data_y = train_data[1]
    valid_data_y = valid_data[1]
    logging.info('mean:{}, std:{}'.format(mean, stddev))

    @tf.function
    def augmentation(x, y, pad=4):
        x = tf.pad(x, [[0, 0], [pad, pad], [pad, pad], [0, 0]])
        x = tf.image.random_crop(x, [x.shape[0], 32, 32, 3])
        x = tf.image.random_flip_left_right(x)
        return x, y

    train_dataset = tf.data.Dataset.from_tensor_slices(
        (train_data_x, train_data_y))
    valid_dataset = tf.data.Dataset.from_tensor_slices(
        (valid_data_x, valid_data_y))
    train_dataset = tf.data.Dataset.from_tensor_slices(
        (train_data_x, train_data_y))
    valid_dataset = tf.data.Dataset.from_tensor_slices(
        (valid_data_x, valid_data_y))
    train_dataset = train_dataset.batch(FLAGS.bs, drop_remainder=True).map(
        augmentation,
        num_parallel_calls=tf.data.experimental.AUTOTUNE).cache().shuffle(
            50000).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    valid_dataset = valid_dataset.batch(
        FLAGS.bs, drop_remainder=False).cache().prefetch(
            buffer_size=tf.data.experimental.AUTOTUNE)
    return [train_dataset, valid_dataset]
