""" OOD (out-of-distribution)
OOD -> y=1
ID -> y=0
"""
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import os
import glob
import copy
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
from absl import flags, logging
from tf_rlib import datasets
from tqdm.auto import tqdm
from tf_rlib.datasets.augmentation import SVDBlur

FLAGS = flags.FLAGS
LOGGER = logging.get_absl_logger()

###########################################
######## RandomNetworkDistillation ########
###########################################

###########################################
############### AutoEncoder ###############
###########################################


class MnistOOD(datasets.Dataset):
    def __init__(self):
        super(MnistOOD, self).__init__()

    def get_data(self, target_idx):
        """
        target_idx(int): ranged from 0~9
        """
        train_data, valid_data = tf.keras.datasets.mnist.load_data()
        train_data_x = train_data[0][..., None].astype(np.float32)
        valid_data_x = valid_data[0][..., None].astype(np.float32)
        train_data_y = train_data[1]
        valid_data_y = valid_data[1]
        # Normalizing the images to the range of [-1., 1.]
        train_data_x = train_data_x / 128. - 1.
        valid_data_x = valid_data_x / 128. - 1.
        # select target
        train_target = train_data_x[np.argwhere(train_data_y == target_idx)[:,
                                                                            0]]
        valid_target = valid_data_x[np.argwhere(valid_data_y == target_idx)[:,
                                                                            0]]
        valid_others = valid_data_x[np.argwhere(valid_data_y != target_idx)[:,
                                                                            0]]

        train_x = train_target
        train_y = np.zeros([len(train_target), 1], np.float32)
        valid_x = np.concatenate([valid_target, valid_others], axis=0)
        valid_y = np.concatenate([
            np.zeros([len(valid_target), 1], np.float32),
            np.ones([len(valid_others), 1], np.float32)
        ],
                                 axis=0)

        train_dataset = tf.data.Dataset.from_tensor_slices((train_x, train_y))
        valid_dataset = tf.data.Dataset.from_tensor_slices((valid_x, valid_y))
        train_dataset = train_dataset.cache().shuffle(len(train_target)).batch(
            FLAGS.bs, drop_remainder=True).prefetch(
                buffer_size=tf.data.experimental.AUTOTUNE)
        valid_dataset = valid_dataset.cache().batch(
            FLAGS.bs, drop_remainder=False).prefetch(
                buffer_size=tf.data.experimental.AUTOTUNE)
        return train_dataset, valid_dataset


class Cifar10vsSVHN(datasets.Dataset):
    """ [Novelty Detection Via Blurring](https://arxiv.org/abs/1911.11943)
    Cifar10:SVHN (Target:OOD)
        train: 50000 Cifar10
        test: 10000 Cifar10 + 10000 SVHN
    """
    def __init__(self):
        super(Cifar10vsSVHN, self).__init__()

    def get_data(self):
        # cifar10: in-distribution
        cifar, info = tfds.load("cifar10:3.0.0",
                                with_info=True,
                                split=['train', 'test'])
        cifar_train, cifar_test = cifar
        # SVHN: out-of-distribution
        svhn, info = tfds.load("svhn_cropped:3.0.0",
                               with_info=True,
                               split=['train', 'test'])
        svhn_train, svhn_test = svhn

        # parse fn
        @tf.function
        def parse_nomal(example):
            x = tf.cast(example['image'], tf.float32)
            x = (x / 128.0) - 1.0
            return x, tf.zeros([
                1,
            ], np.float32)

        @tf.function
        def parse_abnormal(example):
            x = tf.cast(example['image'], tf.float32)
            x = (x / 128.0) - 1.0
            return x, tf.ones([
                1,
            ], np.float32)

        train_dataset = cifar_train.map(
            parse_nomal,
            num_parallel_calls=tf.data.experimental.AUTOTUNE).cache().shuffle(
                50000).batch(FLAGS.bs, drop_remainder=True).prefetch(
                    buffer_size=tf.data.experimental.AUTOTUNE)
        cifar_test = cifar_test.map(
            parse_nomal,
            num_parallel_calls=tf.data.experimental.AUTOTUNE).cache()
        svhn_test = svhn_test.map(
            parse_abnormal,
            num_parallel_calls=tf.data.experimental.AUTOTUNE).cache()
        valid_dataset = cifar_test.concatenate(svhn_test).shuffle(10000).batch(
            FLAGS.bs, drop_remainder=True).prefetch(
                buffer_size=tf.data.experimental.AUTOTUNE)
        return train_dataset, valid_dataset


class SVDBlurCifar10vsSVHN(datasets.Dataset):
    """ [Novelty Detection Via Blurring](https://arxiv.org/abs/1911.11943)
    Cifar10:SVHN (Target:OOD)
        train: 50000 Cifar10
        test: 10000 Cifar10 + 10000 SVHN
    """
    def __init__(self):
        super(SVDBlurCifar10vsSVHN, self).__init__()
        self.svdblur = SVDBlur(singular_shape=[3, 32])

    def get_data(self):
        # cifar10: in-distribution
        cifar, info = tfds.load("cifar10:3.0.0",
                                with_info=True,
                                split=['train', 'test'])
        cifar_train, cifar_test = cifar
        # SVHN: out-of-distribution
        svhn, info = tfds.load("svhn_cropped:3.0.0",
                               with_info=True,
                               split=['train', 'test'])
        svhn_train, svhn_test = svhn

        # parse fn
        @tf.function
        def parse_training(example):
            x = tf.cast(example['image'], tf.float32)
            # SVD blur
            blur_x = self.svdblur.blur(x, remove=FLAGS.svd_remove)
            blur_x = (blur_x / 128.0) - 1.0
            # vanilla
            x = (x / 128.0) - 1.0
            return (x, blur_x), tf.zeros([
                1,
            ], np.float32)

        @tf.function
        def parse_nomal(example):
            x = tf.cast(example['image'], tf.float32)
            x = (x / 128.0) - 1.0
            return x, tf.zeros([
                1,
            ], np.float32)

        @tf.function
        def parse_abnormal(example):
            x = tf.cast(example['image'], tf.float32)
            x = (x / 128.0) - 1.0
            return x, tf.ones([
                1,
            ], np.float32)

        train_dataset = cifar_train.map(
            parse_training,
            num_parallel_calls=tf.data.experimental.AUTOTUNE).cache().shuffle(
                50000).batch(FLAGS.bs, drop_remainder=True).prefetch(
                    buffer_size=tf.data.experimental.AUTOTUNE)
        cifar_test = cifar_test.map(
            parse_nomal,
            num_parallel_calls=tf.data.experimental.AUTOTUNE).cache()
        svhn_test = svhn_test.map(
            parse_abnormal,
            num_parallel_calls=tf.data.experimental.AUTOTUNE).cache()
        valid_dataset = cifar_test.concatenate(svhn_test).shuffle(10000).batch(
            FLAGS.bs, drop_remainder=True).prefetch(
                buffer_size=tf.data.experimental.AUTOTUNE)
        return train_dataset, valid_dataset


class MVTecDS(datasets.Dataset):
    CSV_NAME = 'all.csv'
    TRAIN_OK_NAME = 'train_ok.npy'
    TEST_OK_NAME = 'test_ok.npy'
    TEST_NG_NAME = 'test_ng.npy'
    TEST_GT_NAME = 'test_gt.npy'
    """
    MVTec_AD CVPR 2019: https://www.mvtec.com/company/research/datasets/mvtec-ad/
    Data Download link: ftp://guest:GU.205dldo@ftp.softronics.ch/mvtec_anomaly_detection/mvtec_anomaly_detection.tar.xz
    """
    def __init__(self, data_path='/ws_data/MVTec_AD', force_update=False):
        """
        @Params
            data_path(str): data path
        """
        super(MVTecDS, self).__init__()
        self.data_path = data_path
        self.df = self.get_df(force_update)

    def get_df(self, force_update):
        df_path = os.path.join(self.data_path, MVTecDS.CSV_NAME)
        df = pd.DataFrame()
        if not os.path.exists(df_path) or force_update:
            LOGGER.info('start to parse MVTec_AD dataset')
            for img_path in tqdm(
                    glob.glob(os.path.join(self.data_path, '*/*/*/*.png'))):
                # examples: /ws_data/MVTec_AD/zipper/train/good/182.png
                name_splits = img_path.split('/')
                name = name_splits[-1]
                type_ = name_splits[-2]
                group = name_splits[-3]
                category = name_splits[-4]
                mask_path = ""
                if group == 'ground_truth':
                    continue
                else:
                    if group == 'test' and type_ != 'good':
                        mask_path = img_path.replace(group,
                                                     'ground_truth').replace(
                                                         '.png', '_mask.png')
                    row = pd.Series({
                        'name': name,
                        'type': type_,
                        'group': group,
                        'category': category,
                        'path': img_path,
                        'mask_path': mask_path
                    })
                    df = df.append(row, ignore_index=True)
            df.to_csv(df_path)
        else:
            df = pd.read_csv(df_path)
        return df

    def get_data(self, category, target_size=(128, 128), force_update=False):
        """ 
        NOTE: default resized by PIL.Image.BILINEAR
        @Params
            category(str): category, such as toothbrush, bottle ...etc.
        """
        # check key
        if category not in self.df.category.unique():
            raise ValueError('{} not exists in {}'.format(
                category, self.df.category.unique()))

        folder_path = os.path.join(self.data_path, category, 'numpy',
                                   str(target_size))
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        train_ok_path = os.path.join(folder_path, MVTecDS.TRAIN_OK_NAME)
        test_ok_path = os.path.join(folder_path, MVTecDS.TEST_OK_NAME)
        test_ng_path = os.path.join(folder_path, MVTecDS.TEST_NG_NAME)
        test_gt_path = os.path.join(folder_path, MVTecDS.TEST_GT_NAME)

        tmp_df = self.df[self.df.category == category]
        train_ok = []
        if not os.path.exists(train_ok_path) or force_update:
            LOGGER.debug('start to collect train_ok images')
            for img_path in tmp_df[(tmp_df.group == 'train')
                                   & (tmp_df.type == 'good')].path:
                img = Image.open(img_path)
                img = img.resize(target_size, Image.BILINEAR)
                img = np.array(img).astype(np.float32)
                train_ok.append(img)
            train_ok = np.array(train_ok)
            np.save(train_ok_path, train_ok)
        else:
            train_ok = np.load(train_ok_path)
        LOGGER.info('train_ok shape: {}, dtype: {}'.format(
            train_ok.shape, train_ok.dtype))

        test_ok = []
        if not os.path.exists(test_ok_path) or force_update:
            LOGGER.debug('start to collect test_ok images')
            for img_path in tmp_df[(tmp_df.group == 'test')
                                   & (tmp_df.type == 'good')].path:
                img = Image.open(img_path)
                img = img.resize(target_size, Image.BILINEAR)
                img = np.array(img).astype(np.float32)
                test_ok.append(img)
            test_ok = np.array(test_ok)
            np.save(test_ok_path, test_ok)
        else:
            test_ok = np.load(test_ok_path)
        LOGGER.info('test_ok shape: {}, dtype: {}'.format(
            test_ok.shape, test_ok.dtype))

        test_ng = []
        test_gt = []
        if not os.path.exists(test_ng_path) or not os.path.exists(
                test_gt_path) or force_update:
            LOGGER.debug('start to collect test_ng and test_gt images')
            tmp_tmp_df = tmp_df[(tmp_df.group == 'test')
                                & (tmp_df.type != 'good')]
            for img_path, mask_path in zip(tmp_tmp_df.path,
                                           tmp_tmp_df.mask_path):
                # img
                img = Image.open(img_path)
                img = img.resize(target_size, Image.BILINEAR)
                img = np.array(img).astype(np.float32)
                test_ng.append(img)
                # mask
                mask = Image.open(mask_path)
                mask = mask.resize(target_size, Image.BILINEAR)
                mask = np.array(mask).astype(np.float32)
                test_gt.append(mask)
            test_ng = np.array(test_ng)
            np.save(test_ng_path, test_ng)
            test_gt = np.array(test_gt)
            np.save(test_gt_path, test_gt)
        else:
            test_ng = np.load(test_ng_path)
            test_gt = np.load(test_gt_path)
        LOGGER.info('test_ng shape: {}, dtype: {}'.format(
            test_ng.shape, test_ng.dtype))
        LOGGER.info('test_gt shape: {}, dtype: {}'.format(
            test_gt.shape, test_gt.dtype))

        LOGGER.info('Normalize to [-1, 1]')
        min_ = np.amin(train_ok)
        range_ = np.amax(train_ok) - min_
        train_ok = (train_ok - min_) / range_ * 2.0 - 1.0
        test_ok = (test_ok - min_) / range_ * 2.0 - 1.0
        test_ng = (test_ng - min_) / range_ * 2.0 - 1.0

        @tf.function
        def augmentation(x,
                         y,
                         pad=int((target_size[0] + target_size[1]) / 2 // 8)):
            x = tf.image.resize_with_crop_or_pad(x, target_size[0] + pad * 2,
                                                 target_size[1] + pad * 2)
            x = tf.image.random_crop(x, target_size + (3, ))
            x = tf.image.random_flip_left_right(x)
            return x, y

        train_ok_ds = tf.data.Dataset.from_tensor_slices(
            (train_ok, tf.ones([
                len(train_ok),
            ], dtype=tf.float32)))
        # validation set = test_ok + test_ng
        test_okng = np.concatenate([test_ok, test_ng], axis=0)
        test_y = np.concatenate([
            tf.ones([
                len(test_ok),
            ], dtype=tf.float32),
            tf.zeros([
                len(test_ng),
            ], dtype=tf.float32)
        ],
                                axis=0)
        test_okng_ds = tf.data.Dataset.from_tensor_slices((test_okng, test_y))

        train_ok_ds = train_ok_ds.map(
            augmentation,
            num_parallel_calls=tf.data.experimental.AUTOTUNE).cache().shuffle(
                train_ok.shape[0]).batch(
                    FLAGS.bs, drop_remainder=True).prefetch(
                        buffer_size=tf.data.experimental.AUTOTUNE)
        test_okng_ds = test_okng_ds.cache().batch(
            FLAGS.bs, drop_remainder=False).prefetch(
                buffer_size=tf.data.experimental.AUTOTUNE)

        return [train_ok_ds, test_okng_ds]


###########################################
################ OneClass #################
###########################################


class Cifar10OneClass(datasets.Dataset):
    def __init__(self):
        super(Cifar10OneClass, self).__init__()

    def get_data(self, target_idx):
        return self._get_dsets(target_idx)

    def _get_dsets(self, target_idx):
        """
        target_idx(int): ranged from 0~9
        """
        train_data, valid_data = tf.keras.datasets.cifar10.load_data()
        train_data_x = train_data[0].astype(np.float32)
        valid_data_x = valid_data[0].astype(np.float32)
        mean = train_data_x.mean(axis=(0, 1, 2))
        stddev = train_data_x.std(axis=(0, 1, 2))
        train_data_x = (train_data_x - mean) / stddev
        valid_data_x = (valid_data_x - mean) / stddev
        train_data_y = train_data[1]
        valid_data_y = valid_data[1]
        LOGGER.info('mean:{}, std:{}'.format(mean, stddev))

        # select target
        train_target = train_data_x[np.argwhere(train_data_y == target_idx)[:,
                                                                            0]]
        valid_target = valid_data_x[np.argwhere(valid_data_y == target_idx)[:,
                                                                            0]]
        #         train_others = valid_data_x[np.argwhere(train_data_y != target_idx)[:,0]]
        valid_others = valid_data_x[np.argwhere(valid_data_y != target_idx)[:,
                                                                            0]]

        train_x = train_target
        train_y = np.zeros([len(train_target), 1], np.float32)
        valid_x = np.concatenate([valid_target, valid_others], axis=0)
        valid_y = np.concatenate([
            np.zeros([len(valid_target), 1], np.float32),
            np.ones([len(valid_others), 1], np.float32)
        ],
                                 axis=0)

        @tf.function
        def augmentation(x, y, pad=4):
            x = tf.image.resize_with_crop_or_pad(x, 32 + pad * 2, 32 + pad * 2)
            x = tf.image.random_crop(x, [32, 32, 3])
            x = tf.image.random_flip_left_right(x)
            return x, y

        train_dataset = tf.data.Dataset.from_tensor_slices((train_x, train_y))
        valid_dataset = tf.data.Dataset.from_tensor_slices((valid_x, valid_y))
        train_dataset = train_dataset.cache().map(
            augmentation,
            num_parallel_calls=tf.data.experimental.AUTOTUNE).shuffle(
                len(train_target)).batch(
                    FLAGS.bs, drop_remainder=True).prefetch(
                        buffer_size=tf.data.experimental.AUTOTUNE)
        valid_dataset = valid_dataset.cache().batch(
            FLAGS.bs, drop_remainder=False).prefetch(
                buffer_size=tf.data.experimental.AUTOTUNE)
        return [train_dataset, valid_dataset]


class UDMAA02FD(datasets.Dataset):
    """ Used for one-class/active-authentication research.
    https://umdaa02.github.io/, please download, unzip and save UDMAA02-FD dataset to '/ws_data/UMDAA-02/Data/'.
    
    number of images: 33209
    average images per user: 755
    
    """
    def __init__(self, root_path='/ws_data/UMDAA-02/Data/'):
        """
        root_path(str): 
        """
        super(UDMAA02FD, self).__init__()
        self.root_path = root_path

    def get_data(self, target_idx, train_ratio=0.8, input_shape=(128, 128)):
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
