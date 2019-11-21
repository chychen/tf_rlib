import tensorflow as tf
import numpy as np
import os
import glob
import pandas as pd
from tqdm.auto import tqdm
from absl import flags, logging
from PIL import Image

FLAGS = flags.FLAGS
LOGGER = logging.get_absl_logger()


class MVTecDS:
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

        # Normalize
        mean = train_ok.mean(axis=(0, 1, 2))
        stddev = train_ok.std(axis=(0, 1, 2))
        LOGGER.info('mean:{}, std:{}'.format(mean, stddev))
        train_ok = (train_ok - mean) / stddev
        test_ok = (test_ok - mean) / stddev
        test_ng = (test_ng - mean) / stddev

        @tf.function
        def augmentation(x,
                         y,
                         pad=int((target_size[0] + target_size[1]) / 2 // 8)):
            x = tf.image.resize_with_crop_or_pad(x, target_size[0] + pad * 2,
                                                 target_size[1] + pad * 2)
            x = tf.image.random_crop(x, target_size + (3, ))
            x = tf.image.random_flip_left_right(x)
            return x, y

        train_ok_ds = tf.data.Dataset.from_tensor_slices((train_ok, train_ok))
        test_ok_ds = tf.data.Dataset.from_tensor_slices((test_ok, test_ok))
        # test_ok + test_ng
        test_okng = np.concatenate([test_ok, test_ng], axis=0)
        #         shuffle_idx = np.random.shuffle(np.arange(len(test_okng)))
        #         test_okng = test_okng[shuffle_idx]
        test_okng_ds = tf.data.Dataset.from_tensor_slices(
            (test_okng, test_okng))

        train_ok_ds = train_ok_ds.map(
            augmentation,
            num_parallel_calls=tf.data.experimental.AUTOTUNE).cache().shuffle(
                train_ok.shape[0]).batch(
                    FLAGS.bs, drop_remainder=True).prefetch(
                        buffer_size=tf.data.experimental.AUTOTUNE)
        test_ok_ds = test_ok_ds.cache().batch(
            FLAGS.bs, drop_remainder=False).prefetch(
                buffer_size=tf.data.experimental.AUTOTUNE)
        test_okng_ds = test_okng_ds.cache().batch(
            FLAGS.bs, drop_remainder=False).prefetch(
                buffer_size=tf.data.experimental.AUTOTUNE)
        return [train_ok_ds, test_ok_ds, test_okng_ds]
