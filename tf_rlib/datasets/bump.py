import tensorflow as tf
from tensorflow.keras.utils import to_categorical
import numpy as np
from tf_rlib import datasets
from absl import flags, logging

FLAGS = flags.FLAGS
LOGGER = logging.get_absl_logger()


class NVBump(datasets.Dataset):
    '''
    Mount point: Warren_datasets/SPIL/data/preprocessed/
    '''
    def __init__(self, path='/mount/data/SPIL/data/preprocessed/'):
        self.path = path
        super(NVBump, self).__init__()
        self.dsets = self._get_dsets()

    def get_data(self):
        return self.dsets

    def _get_dsets(self):
        # load data
        X_defect = np.load(self.path + 'X_defect.npy')[:, 172:428, 172:428]
        X_pass = np.load(self.path + 'X_pass.npy')[:, 172:428, 172:428]
        X_defect_valid = np.load(self.path + 'X_defect_val.npy')
        X_pass_valid = np.load(self.path + 'X_pass_val.npy')
        # Spilt more data to be validated
        idx = np.arange(len(X_defect))
        np.random.shuffle(idx)
        X_defect_valid = np.concatenate([X_defect_valid, X_defect[:305]])
        X_defect = X_defect[305:]
        # get generator
        gen_train = self.get_generator(X_defect, X_pass)
        gen_valid = self.get_generator(X_defect_valid, X_pass_valid)
        # tf dataset
        train_dataset = tf.data.Dataset.from_generator(
            gen_train, (tf.float32, tf.float32),
            (tf.TensorShape([None, 224, 224, 3]), tf.TensorShape([None])))
        valid_dataset = tf.data.Dataset.from_generator(
            gen_valid, (tf.float32, tf.float32),
            (tf.TensorShape([None, 224, 224, 3]), tf.TensorShape([None])))
        train_dataset = train_dataset.cache().shuffle(2000).prefetch(
            buffer_size=tf.data.experimental.AUTOTUNE)
        valid_dataset = valid_dataset.cache().prefetch(
            buffer_size=tf.data.experimental.AUTOTUNE)
        return [train_dataset, valid_dataset]

    def random_crop(self, img, random_crop_size):
        height, width = img.shape[0], img.shape[1]
        dy, dx = random_crop_size
        x = np.random.randint(0, width - dx + 1)
        y = np.random.randint(0, height - dy + 1)
        return img[y:(y + dy), x:(x + dx), :]

    def augment(self, X):
        # flip
        toflip = np.random.uniform()
        if toflip > .67:
            X = np.flip(X, axis=0)
        elif toflip > .33:
            X = np.flip(X, axis=1)
        X = self.random_crop(X, (224, 224))
        X = tf.keras.preprocessing.image.random_brightness(X, (.8, 1.2))
        return X

    def sample_data(self, X_d, X_p, idx_d, idx_p):
        np.random.shuffle(idx_d)
        np.random.shuffle(idx_p)
        x, y = [], []
        for i in range(FLAGS.bs // 2):
            xp = X_p[idx_p[i]]
            xd = X_d[idx_d[i]]
            xp = (self.augment(xp) - 127.5) / 127.5
            xd = (self.augment(xd) - 127.5) / 127.5
            x += [xp, xd]
            y += [0, 1]
        return np.array(x), np.array(y)

    def get_generator(self, X_d, X_p):
        def gen():
            idx_d = np.arange(len(X_d))
            idx_p = np.arange(len(X_p))
            for _ in range(len(X_d) // FLAGS.bs):
                x, y = self.sample_data(X_d, X_p, idx_d, idx_p)
                yield x, y.astype('float32')

        return gen
