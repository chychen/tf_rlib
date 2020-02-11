import os
import glob
import copy
import numpy as np
import tensorflow as tf
from tqdm.auto import tqdm
from absl import flags, logging
from tf_rlib import datasets

FLAGS = flags.FLAGS
LOGGER = logging.get_absl_logger()


class UDMAA02(datasets.Dataset):
    """ Used for one-class/active-authentication research.
    https://umdaa02.github.io/, please download, unzip and save UDMAA02-FD dataset to '/ws_data/UMDAA-02/Data/'.
    
    number of images: 33209
    average images per user: 755
    
    """
    def __init__(self, root_path='/ws_data/UMDAA-02/Data/'):
        """
        root_path(str): 
        """
        super(UDMAA02, self).__init__()
        self.root_path = root_path

    def get_data(self, target_idx=0, train_ratio=0.8, input_shape=(128, 128)):
        """
        target_idx(int): ranged from 0~43
        """
        users = glob.glob(os.path.join(self.root_path, '*'))
        target_user = copy.deepcopy(users[target_idx])
        users.remove(target_user)
        other_users = copy.deepcopy(users)

        target_imgs = glob.glob(
            os.path.join(target_user, '*/*/*/*/*/*/*/*.jpg'))
        train_len = int(len(target_imgs) * train_ratio)
        train_target_imgs = target_imgs[:train_len]
        valid_target_imgs = target_imgs[train_len:]

        valid_others_imgs = []
        for other_user in other_users[:]:
            other_imgs = glob.glob(
                os.path.join(other_user, '*/*/*/*/*/*/*/*.jpg'))
            valid_len = int(len(other_imgs) * train_ratio)
            valid_others_imgs = valid_others_imgs + other_imgs[valid_len:]

        @tf.function
        def parse_image(filename, y, pad=input_shape[0] // 8):
            image = tf.io.read_file(filename)
            image = tf.image.decode_jpeg(image)
            image = tf.image.convert_image_dtype(image, tf.float32)
            image = tf.image.resize(image, input_shape)
            # augmentation
            image = tf.image.resize_with_crop_or_pad(image,
                                                     input_shape[0] + pad * 2,
                                                     input_shape[1] + pad * 2)
            image = tf.image.random_crop(image, input_shape + (3, ))
            image = tf.image.random_flip_left_right(image)
            image = image * 2. - 1.  # norm to (-1, +1)
            return image, y

        # Train
        train_dset = tf.data.Dataset.from_tensor_slices(
            (train_target_imgs,
             np.ones(shape=[len(train_target_imgs), 1], dtype=np.float32)))
        train_dset = train_dset.map(
            parse_image,
            num_parallel_calls=tf.data.experimental.AUTOTUNE).cache().shuffle(
                1000).batch(FLAGS.bs, drop_remainder=True).prefetch(
                    buffer_size=tf.data.experimental.AUTOTUNE)

        # Validation
        valid_dset = tf.data.Dataset.from_tensor_slices(
            (valid_target_imgs,
             np.ones(shape=[len(valid_target_imgs), 1], dtype=np.float32)))
        valid_others_dset = tf.data.Dataset.from_tensor_slices(
            (valid_others_imgs,
             np.zeros(shape=[len(valid_others_imgs), 1], dtype=np.float32)))
        valid_dset = valid_dset.concatenate(valid_others_dset)
        valid_dset = valid_dset.map(
            parse_image,
            num_parallel_calls=tf.data.experimental.AUTOTUNE).cache().batch(
                FLAGS.bs, drop_remainder=True).prefetch(
                    buffer_size=tf.data.experimental.AUTOTUNE)

        return train_dset, valid_dset
