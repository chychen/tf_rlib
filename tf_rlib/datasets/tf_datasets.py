import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import os
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
from absl import flags, logging
from tf_rlib import datasets
from tqdm.auto import tqdm

FLAGS = flags.FLAGS
LOGGER = logging.get_absl_logger()


class Omniglot(datasets.Dataset):
    SAVE_PATH = '/tmp/omniglot/'

    def __init__(self,
                 n_train_episode=5000,
                 n_valid_episode=1000,
                 img_size=(28, 28),
                 force_update=False,
                 n_train_class=1200):
        super(Omniglot, self).__init__()
        self.n_train_episode = n_train_episode
        self.n_valid_episode = n_valid_episode
        self.c_way = FLAGS.c_way
        self.k_shot = FLAGS.k_shot
        self.img_shape = img_size + (1, )
        self.n_train_class = n_train_class
        self.np_dset = self._get_np_dset(img_size, force_update)
        # get mean,std
        self.mean = np.mean(self.np_dset[:self.n_train_class])
        self.std = np.std(self.np_dset[:self.n_train_class])
        LOGGER.info('mean: {}, stddev: {}'.format(self.mean, self.std))
        # normalize
        self.np_dset = (self.np_dset - self.mean) / self.std

    def get_data(self):
        return self._get_tf_dsets(self.np_dset)

    def _get_np_dset(self, img_size, force_update):
        if not os.path.exists(Omniglot.SAVE_PATH):
            os.makedirs(Omniglot.SAVE_PATH)

        dset_path = os.path.join(Omniglot.SAVE_PATH,
                                 'dset_{}.npy'.format(img_size))

        if not os.path.exists(dset_path) or force_update:
            [
                dset_all,
            ], info = tfds.load("omniglot:3.0.0",
                                with_info=True,
                                split=['train+test'])

            def resize(x):
                x['image'] = tf.image.resize(x['image'], size=img_size)
                x['image'] = tf.image.rgb_to_grayscale(x['image'])
                return x

            dset_all = dset_all.map(resize)
            np_dset = np.zeros(
                shape=[info.features['label'].num_classes, 20, 28, 28, 1],
                dtype=np.float32)
            counter = np.zeros(shape=[
                info.features['label'].num_classes,
            ],
                               dtype=np.int32)
            for i, v in enumerate(tqdm(dset_all)):
                c_idx = v['label'].numpy()
                np_dset[c_idx, counter[c_idx]] = v['image'].numpy()
                counter[c_idx] += 1

            np.save(dset_path, np_dset)
        else:
            np_dset = np.load(dset_path)

        return np_dset

    def _get_tf_dsets(self, np_dset):
        train_dataset = tf.data.Dataset.from_generator(
            lambda: self._episode_generator(np_dset,
                                            self.n_train_episode,
                                            True,
                                            c_way=self.c_way,
                                            k_shot=self.k_shot,
                                            img_shape=self.img_shape),
            output_types=((np.float32, np.float32),
                          np.float32)).cache().shuffle(10000).batch(
                              FLAGS.bs, drop_remainder=True).prefetch(
                                  buffer_size=tf.data.experimental.AUTOTUNE)
        valid_dataset = tf.data.Dataset.from_generator(
            lambda: self._episode_generator(np_dset,
                                            self.n_valid_episode,
                                            False,
                                            c_way=self.c_way,
                                            k_shot=self.k_shot,
                                            img_shape=self.img_shape),
            output_types=((np.float32, np.float32), np.float32)).cache().batch(
                FLAGS.bs, drop_remainder=False).prefetch(
                    buffer_size=tf.data.experimental.AUTOTUNE)

        return (train_dataset, valid_dataset)

    def _episode_generator(self,
                           np_dset,
                           n_episode,
                           is_train,
                           c_way=5,
                           k_shot=1,
                           img_shape=[28, 28, 1]):
        n_samples = np_dset.shape[1]  # n_samples=20
        if is_train:
            pool = list(range(0, self.n_train_class))
            n_query_per_c = (n_samples - k_shot)
            n_query = n_query_per_c * c_way  # n_query=19*5
        else:
            pool = list(range(self.n_train_class, np_dset.shape[0]))
            n_query_per_c = 1
            n_query = n_query_per_c * c_way  # n_query=1*5

        for i in range(n_episode):
            support_set = np.zeros(shape=[n_query, c_way, k_shot, *img_shape],
                                   dtype=np.float32)
            query_set = np.zeros(shape=[n_query, c_way, k_shot, *img_shape],
                                 dtype=np.float32)
            y = np.zeros(shape=[n_query, c_way], dtype=np.float32)

            selected_c_idx = np.random.choice(pool, c_way, replace=False)
            selected_c_data = np_dset[selected_c_idx]
            # rotation augment classes
            rotate_degree = np.random.randint(low=0, high=4, size=c_way)
            for i, v in enumerate(rotate_degree):
                selected_c_data[i] = np.rot90(selected_c_data[i],
                                              v,
                                              axes=(-3, -2))

            for c in range(c_way):
                perm_idx = np.random.permutation(n_samples)
                selected_s_idx = perm_idx[:k_shot]
                # support set
                selected_s_per_query = selected_c_data[
                    c, selected_s_idx]  # k_shot data with same class
                support_set[:, c:c + 1] = np.tile(selected_s_per_query,
                                                  [n_query, 1, 1, 1, 1, 1])
                # query set
                selected_q_idx = perm_idx[k_shot:k_shot +
                                          n_query_per_c]  # n_query_per_c
                selected_q = selected_c_data[
                    c, selected_q_idx]  # n_query data with same class
                tmp_q = np.repeat(selected_q, k_shot * c_way, axis=0)
                tmp_q = np.reshape(tmp_q,
                                   [n_query_per_c, c_way, k_shot, *img_shape])
                query_set[c * n_query_per_c:(c + 1) * n_query_per_c] = tmp_q
                y[c * n_query_per_c:(c + 1) * n_query_per_c,
                  c] = np.ones(shape=[
                      n_query_per_c,
                  ], dtype=np.float32)

            yield (support_set, query_set), y

    def vis(self, num_samples=10):
        tfdset = self.data[0]  # train
        v = next(iter(tfdset))
        s = v[0][0][0]  # tuple->support_set->batch
        q = v[0][1][0]  # tuple->query_set->batch
        y = v[1][0]
        sampled_idx = np.random.permutation(q.shape[0])[:num_samples]
        for n in sampled_idx:
            fig = plt.figure(figsize=[6, 1])
            for c in range(q.shape[1]):
                k = 0
                plt.subplot(1, 6, c + 1)
                plt.imshow(s[n, c, k][..., 0], cmap='gray')
            plt.subplot(1, 6, 6)
            plt.title('query:{}'.format(y[n]))
            plt.imshow(q[n, 0, k][..., 0], cmap='gray')


class Cifar10Numpy(datasets.Dataset):
    def __init__(self):
        super(Cifar10Numpy, self).__init__()

    def get_data(self):
        return self._get_dsets()

    def _get_dsets(self):
        train_data, valid_data = tf.keras.datasets.cifar10.load_data()
        train_data_x = train_data[0].astype(np.float32)
        valid_data_x = valid_data[0].astype(np.float32)
        mean = train_data_x.mean(axis=(0, 1, 2))
        stddev = train_data_x.std(axis=(0, 1, 2))
        train_data_x = (train_data_x - mean) / stddev
        valid_data_x = (valid_data_x - mean) / stddev
        if FLAGS.amp:
            train_data_x = train_data_x.astype(np.float16)
            valid_data_x = valid_data_x.astype(np.float16)
        train_data_y = train_data[1]
        valid_data_y = valid_data[1]
        LOGGER.info('mean:{}, std:{}'.format(mean, stddev))

        @tf.function
        def augmentation(x, y, pad=4):
            x = tf.cast(x, tf.float32)
            x = (x - mean) / stddev
            x = tf.image.resize_with_crop_or_pad(x, 32 + pad * 2, 32 + pad * 2)
            x = tf.image.random_crop(x, [32, 32, 3])
            x = tf.image.random_flip_left_right(x)
            return x, y

        @tf.function
        def normalize(x, y, pad=4):
            x = tf.cast(x, tf.float32)
            x = (x - mean) / stddev
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

    # batchwise is slower in this augmentation
    @tf.function
    def augmentation(x, y, pad=4):
        bs = tf.shape(x)[0]
        cropped_x = tf.pad(x, [[0, 0], [pad, pad], [pad, pad], [0, 0]])
        cropped_x = tf.image.random_crop(x, [bs, 32, 32, 3])
        choice_crop = tf.random.uniform(shape=[bs, 1, 1, 1],
                                        minval=0.0,
                                        maxval=1.0)
        x = tf.where(tf.less(choice_crop, 0.5), cropped_x, x)
        choice_flip = tf.random.uniform(shape=[bs, 1, 1, 1],
                                        minval=0.0,
                                        maxval=1.0)
        x = tf.where(tf.less(choice_flip, 0.5),
                     tf.image.random_flip_left_right(x), x)
        return x, y

class Cifar10(datasets.Dataset):
    """ use tfrecords could speed-up 5%~15% comparing to numpy. because tfrecords format only use 50% space than numpy.
    """
    def __init__(self):
        super(Cifar10, self).__init__()
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
        def augmentation(example_proto, pad=4):
            example = tf.io.parse_single_example(example_proto,
                                                 image_feature_description)
            x = tf.io.decode_raw(example['image_raw'], self.dtype)
            x = tf.reshape(x, [32, 32, 3])
            x = tf.image.resize_with_crop_or_pad(x, 32 + pad * 2, 32 + pad * 2)
            x = tf.image.random_crop(x, [32, 32, 3])
            x = tf.image.random_flip_left_right(x)
            return x, example['label']

        @tf.function
        def parse(example_proto):
            example = tf.io.parse_single_example(example_proto,
                                                 image_feature_description)
            x = tf.io.decode_raw(example['image_raw'], self.dtype)
            x = tf.reshape(x, [32, 32, 3])
            return x, example['label']

        train_dataset = tf.data.TFRecordDataset(self.train_file)
        train_dataset = train_dataset.map(
            augmentation,
            num_parallel_calls=tf.data.experimental.AUTOTUNE).cache().shuffle(
                50000).batch(FLAGS.bs, drop_remainder=True).prefetch(
                    buffer_size=tf.data.experimental.AUTOTUNE)

        valid_dataset = tf.data.TFRecordDataset(self.valid_file)
        valid_dataset = valid_dataset.map(
            parse,
            num_parallel_calls=tf.data.experimental.AUTOTUNE).cache().batch(
                FLAGS.bs, drop_remainder=False).prefetch(
                    buffer_size=tf.data.experimental.AUTOTUNE)
        return [train_dataset, valid_dataset]
