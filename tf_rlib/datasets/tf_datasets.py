import tensorflow as tf
from tensorflow.keras.utils import to_categorical
import numpy as np
from absl import flags, logging

FLAGS = flags.FLAGS


def get_cifar10():
    train_data, valid_data = tf.keras.datasets.cifar10.load_data()
    train_data_x = train_data[0].astype(np.float32)
    valid_data_x = valid_data[0].astype(np.float32)
    mean = train_data_x.mean(axis=(0, 1, 2))
    stddev = train_data_x.std(axis=(0, 1, 2))
    train_data_x = (train_data_x - mean) / stddev
    valid_data_x = (valid_data_x - mean) / stddev
    train_data_y = train_data[1]
    valid_data_y = valid_data[1]
    logging.info('mean:{}, std:{}'.format(mean, stddev))

    @tf.function
    def augmentation(x, y, pad=4):
        x = tf.image.resize_with_crop_or_pad(x, 32 + pad * 2, 32 + pad * 2)
        x = tf.image.random_crop(x, [32, 32, 3])
        x = tf.image.random_flip_left_right(x)
        return x, y

    train_dataset = tf.data.Dataset.from_tensor_slices(
        (train_data_x, train_data_y))
    valid_dataset = tf.data.Dataset.from_tensor_slices(
        (valid_data_x, valid_data_y))
    train_dataset = train_dataset.map(
        augmentation,
        num_parallel_calls=tf.data.experimental.AUTOTUNE).cache().shuffle(
            50000).batch(FLAGS.bs, drop_remainder=True).prefetch(
                buffer_size=tf.data.experimental.AUTOTUNE)
    valid_dataset = valid_dataset.cache().batch(
        FLAGS.bs, drop_remainder=False).prefetch(
            buffer_size=tf.data.experimental.AUTOTUNE)
    return [train_dataset, valid_dataset]


#     # batchwise is slower in this augmentation
#     @tf.function
#     def augmentation(x, y, pad=4):
#         bs = tf.shape(x)[0]
#         cropped_x = tf.pad(x, [[0, 0], [pad, pad], [pad, pad], [0, 0]])
#         cropped_x = tf.image.random_crop(x, [bs, 32, 32, 3])
#         choice_crop = tf.random.uniform(shape=[bs, 1, 1, 1], minval=0.0, maxval=1.0)
#         x = tf.where(tf.less(choice_crop, 0.5), cropped_x, x)
#         choice_flip = tf.random.uniform(shape=[bs, 1, 1, 1], minval=0.0, maxval=1.0)
#         x = tf.where(tf.less(choice_flip, 0.5), tf.image.random_flip_left_right(x), x)
#         return x, y


def get_cell(path='/mount/data/SegBenchmark/medical/cell/'):
    X = np.load(path + 'train/X.npy')
    Y = np.load(path + 'train/Y.npy')[..., None]
    Y = to_categorical(Y, 2)
    train_data_x, valid_data_x = X[:int(len(X) * .8)], X[int(len(X) * .8):]
    train_data_y, valid_data_y = Y[:int(len(X) * .8)], Y[int(len(X) * .8):]

    mean = train_data_x.mean(axis=(0, 1, 2))
    stddev = train_data_x.std(axis=(0, 1, 2))
    train_data_x = (train_data_x - mean) / stddev
    valid_data_x = (valid_data_x - mean) / stddev
    logging.info('mean:{}, std:{}'.format(mean, stddev))
    logging.info('data size:{}, label size"{}'.format(train_data_x.shape,
                                                      train_data_y.shape))

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
