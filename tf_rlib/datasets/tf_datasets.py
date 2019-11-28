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
                 n_train_episode=10000,
                 n_valid_episode=1000,
                 c_way=5,
                 k_shot=1,
                 img_size=(28, 28),
                 force_update=False):
        super(Omniglot, self).__init__()
        self.n_train_episode = n_train_episode
        self.n_valid_episode = n_valid_episode
        self.c_way = c_way
        self.k_shot = k_shot
        self.img_shape = img_size + (1, )
        self.np_dset = self._get_np_dset(img_size, force_update)
        self.data = self._get_tf_dsets(self.np_dset)

    def get_data(self):
        return self.data

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
            output_types=(np.float32, np.float32, np.int32))
        valid_dataset = tf.data.Dataset.from_generator(
            lambda: self._episode_generator(np_dset,
                                            self.n_valid_episode,
                                            False,
                                            c_way=self.c_way,
                                            k_shot=self.k_shot,
                                            img_shape=self.img_shape),
            output_types=(np.float32, np.float32, np.int32))

        return (train_dataset, valid_dataset)

    def _episode_generator(self,
                           np_dset,
                           n_episode,
                           is_train,
                           c_way=5,
                           k_shot=1,
                           img_shape=[28, 28, 1]):
        n_samples = np_dset.shape[1]  # n_samples=20
        n_query = n_samples - k_shot  # n_query=19
        if is_train:
            pool = list(range(0, 1200))
        else:
            pool = list(range(1200, np_dset.shape[0]))

        for i in range(n_episode):
            support_set = np.zeros(shape=[n_query, c_way, k_shot, *img_shape],
                                   dtype=np.float32)
            query_set = np.zeros(shape=[n_query, c_way, k_shot, *img_shape],
                                 dtype=np.float32)
            y = np.zeros(shape=[n_query, c_way], dtype=np.int32)

            selected_c_idx = np.random.choice(pool, c_way, replace=False)
            selected_c_data = np_dset[selected_c_idx]

            for c in range(c_way):
                perm_idx = np.random.permutation(n_samples)
                # support set
                selected_s_idx = perm_idx[:k_shot]
                selected_s_per_query = selected_c_data[
                    c, selected_s_idx]  # k_shot data with same class
                support_set[:, c:c + 1] = np.tile(selected_s_per_query,
                                                  [n_query, 1, 1, 1, 1, 1])
                # query set
                selected_q_idx = perm_idx[k_shot:]
                selected_q = selected_c_data[
                    c, selected_q_idx]  # n_query data with same class
                tmp_q = np.tile(selected_q, [k_shot, 1, 1, 1, 1, 1])
                tmp_q = np.reshape(tmp_q, [n_query, 1, k_shot, *img_shape])
                query_set[:, c:c + 1] = tmp_q
                # label
                y[:, c] = np.ones(shape=[
                    n_query,
                ], dtype=np.int32) * c

            yield support_set, query_set, y

    def vis(self, num_samples):
        iter_tfdset = iter(self.data[0])
        counter = 0
        for i, (s, q, y) in enumerate(iter_tfdset):
            for n in range(q.shape[0]):
                for c in range(q.shape[1]):
                    for k in range(q.shape[2]):
                        if counter >= num_samples:
                            break
                        else:
                            fig = plt.figure(figsize=[6, 1])
                            for w in range(q.shape[1]):  # c-way
                                plt.subplot(1, 6, w + 1)
                                plt.imshow(s[n, w, k][..., 0], cmap='gray')
                            plt.subplot(1, 6, 6)
                            plt.title('query:{}'.format(y[n, c]))
                            plt.imshow(q[n, c, k][..., 0], cmap='gray')
                            counter += 1


class Cifar10(datasets.Dataset):
    def __init__(self):
        super(Cifar10, self).__init__()
        self.tf_dsets = self._get_dsets()

    def get_data(self):
        return self.tf_dsets

    def _get_dsets(self):
        train_data, valid_data = tf.keras.datasets.cifar10.load_data()
        train_data_x = train_data[0].astype(np.float32)
        valid_data_x = valid_data[0].astype(np.float32)
        mean = train_data_x.mean(axis=(0, 1, 2))
        stddev = train_data_x.std(axis=(0, 1, 2))
        train_data_x = (train_data_x - mean) / stddev
        valid_data_x = (valid_data_x - mean) / stddev
        #     train_data_x = train_data_x.astype(np.float16)
        #     valid_data_x = valid_data_x.astype(np.float16)
        train_data_y = train_data[1]
        valid_data_y = valid_data[1]
        LOGGER.info('mean:{}, std:{}'.format(mean, stddev))

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


# TODO: @warren please help to rewrite the function obeying the template class datasets.Dataset
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
