import tensorflow as tf
import numpy as np
import os
import glob
import pandas as pd
from tqdm.auto import tqdm
from absl import flags, logging
from tf_rlib import datasets

FLAGS = flags.FLAGS
LOGGER = logging.get_absl_logger()


class PHM2018(datasets.Dataset):
    """
    PHM Data Challenge 2018: https://www.phmsociety.org/events/conference/phm/18/data-challenge
    Data Download link: https://drive.google.com/file/d/15Jx9Scq9FqpIGn8jbAQB_lcHSXvIoPzb/view
    """
    LABELS = [
        'TTF_FlowCool Pressure Dropped Below Limit',
        'TTF_Flowcool Pressure Too High Check Flowcool Pump',
        'TTF_Flowcool leak'
    ]

    def __init__(self,
                 target_label,
                 root_path='/ws_data/PHM2018/phm_data_challenge_2018/',
                 tool_id='01_M01',
                 use_categorical_data=True,
                 WINSIZE=50,
                 DOWNSAMPLE=1,
                 NONOVERLAP=5,
                 TRAIN_RATIO=0.7):
        super(PHM2018, self).__init__()
        """
        note: TTF (Time To Failure)
        """
        if target_label not in PHM2018.LABELS:
            raise ValueError('please choose one within {}'.format(
                PHM2018.LABELS))
        self.target_label = target_label
        self.root_path = root_path
        self.tool_id = tool_id
        self.use_categorical_data = use_categorical_data
        self.FAULT_PATH = os.path.join(
            root_path, 'train/train_faults',
            '{}_train_fault_data.csv'.format(tool_id))
        self.DATA_PATH = os.path.join(root_path, 'train',
                                      '{}_DC_train.csv'.format(tool_id))
        self.TTF_PATH = os.path.join(root_path, 'train/train_ttf',
                                     '{}_DC_train.csv'.format(tool_id))

        self.WINSIZE = WINSIZE
        self.DOWNSAMPLE = DOWNSAMPLE
        self.NONOVERLAP = NONOVERLAP
        self.TRAIN_RATIO = TRAIN_RATIO

        self.data, self.dfs, self.mean, self.std, self.max_y = self._get_dsets(
        )

    def _get_dsets(self):
        # read x and y dataframe from csv
        df_data = pd.read_csv(self.DATA_PATH)
        df_ttf = pd.read_csv(self.TTF_PATH)
        # merge and clean up
        all_df = pd.merge(df_data, df_ttf, how='inner', on='time')
        all_df.dropna(axis=0, inplace=True, subset=[
            self.target_label,
        ])
        ROW_INTERVAL = all_df.time.diff().mode().values[0]
        # split into train and valid
        total_len = all_df.values.shape[0]
        all_df.drop(columns='Tool',
                    inplace=True)  # Tool are same within same machine
        target_label_idx = all_df.columns.get_loc(self.target_label)
        all_df = all_df.astype('float32')
        train_df = all_df.iloc[:int(total_len *
                                    self.TRAIN_RATIO)].reset_index(drop=True)
        desc = train_df.describe().astype('float32')
        mean_df = desc.T['mean'][:-3]
        std_df = desc.T['std'][:-3]
        y_max_df = desc.T['max'][target_label_idx:target_label_idx + 1]
        LOGGER.info('X mean\n{}\nstd:\n{}'.format(mean_df, std_df))
        LOGGER.info('y max:\n{}'.format(y_max_df))
        mean = mean_df.values
        std = mean_df.values
        y_max = y_max_df.values
        valid_df = all_df.iloc[int(total_len *
                                   self.TRAIN_RATIO):].reset_index(drop=True)

        def wrap_as_tfdataset(df):
            segment_idx_df = df[(df.time.diff() != ROW_INTERVAL)]
            segment_idx = segment_idx_df.index.values
            # windowing x
            start = 0
            result_x = []
            result_y = []
            data_x = df.values[:, :-3]
            data_y = df.values[:, target_label_idx:target_label_idx + 1]
            data_x = (data_x - mean) / std
            data_y = data_y / y_max
            for end in tqdm(segment_idx[1:]):
                tmp_x = data_x[start:end]
                tmp_y = data_y[start:end]
                if tmp_x.shape[
                        0] > self.WINSIZE * self.DOWNSAMPLE:  # big enough
                    tmp_result_x = []
                    # -1 because x(t-k)~x(t) -> y(t)
                    for i in range(
                            0, tmp_x.shape[0] -
                            self.WINSIZE * self.DOWNSAMPLE + 1,
                            self.NONOVERLAP):
                        tmp_result_x.append(
                            tmp_x[i:i + self.WINSIZE *
                                  self.DOWNSAMPLE:self.DOWNSAMPLE])
                    tmp_result_x = np.array(tmp_result_x)
                    tmp_result_y = tmp_y[self.WINSIZE * self.DOWNSAMPLE -
                                         1::self.NONOVERLAP]
                    result_x.append(tmp_result_x)
                    result_y.append(tmp_result_y)
                start = end
            result_x = np.concatenate(result_x, axis=0)
            result_y = np.concatenate(result_y, axis=0)
            dset = tf.data.Dataset.from_tensor_slices((result_x, result_y))
            return dset

        train_dset = wrap_as_tfdataset(train_df)
        train_dset = train_dset.cache().shuffle(100000).batch(
            FLAGS.bs, drop_remainder=True).prefetch(
                buffer_size=tf.data.experimental.AUTOTUNE)
        valid_dset = wrap_as_tfdataset(valid_df)
        valid_dset = valid_dset.cache().batch(
            FLAGS.bs, drop_remainder=False).prefetch(
                buffer_size=tf.data.experimental.AUTOTUNE)

        return (train_dset, valid_dset), (train_df, valid_df), mean, std, y_max

    def get_data(self):
        return self.data

    def vis(self, num_samples):
        raise NotImplementedError


#     def _get_dsets(self):
#         # read x and y dataframe from csv
#         df_data = pd.read_csv(self.DATA_PATH)
#         df_ttf = pd.read_csv(self.TTF_PATH)
#         # merge and clean up
#         all_df = pd.merge(df_data, df_ttf, how='inner', on='time')
#         all_df.dropna(axis=0, inplace=True)
#         ROW_INTERVAL = all_df.time.diff().mode().values[0]
#         # split into train and valid
#         total_len = all_df.values.shape[0]
#         train_df = all_df.iloc[:int(total_len * self.TRAIN_RATIO)]
#         valid_df = all_df.iloc[int(total_len * self.TRAIN_RATIO):]

#         def df_handler(tmp_data):
#             x_dset = tf.data.Dataset.from_tensor_slices(
#                 tmp_data.values[:, :-3])
#             y_dset = tf.data.Dataset.from_tensor_slices(
#                 tmp_data.values[:, -3:])
#             x_dset = x_dset.window(self.WINSIZE,
#                                     shift=self.NONOVERLAP,
#                                     stride=self.DOWNSAMPLE,
#                                     drop_remainder=True)
#             x_dset = x_dset.flat_map(
#                 lambda window: window.batch(self.WINSIZE))
#             tmp_dset = tf.data.Dataset.zip(
#                 (x_dset, y_dset))
#             return tmp_dset

#         def wrap_as_tfdataset(df):
#             df.drop(columns='Tool',
#                     inplace=True)  # Tool are same within same machine
#             df = df.astype('float32')
#             segment_idx_df = df[(df.time.diff() != ROW_INTERVAL)]
#             segment_idx = segment_idx_df.index.values
#             # difine first dset
#             start = 0
#             end = segment_idx[1]
#             tmp_data = df[start:end]
#             all_dsets = df_handler(tmp_data)
#             start = end
#             # use first dset concat with others
#             for end in tqdm(segment_idx[2:]):
#                 tmp_data = df[start:end]
#                 tmp_dset = df_handler(tmp_data)
#                 all_dsets = all_dsets.concatenate(tmp_dset)
#                 start = end
#             return all_dsets

#         train_dset = wrap_as_tfdataset(train_df)
#         train_dset = train_dset.batch(FLAGS.bs, drop_remainder=True)
#         valid_dset = wrap_as_tfdataset(valid_df)
#         valid_dset = valid_dset.batch(FLAGS.bs, drop_remainder=False)

#         return train_dset, valid_dset
