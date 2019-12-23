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
    def __init__(
            self,
            root_path='/ws_data/PHM2018/phm_data_challenge_2018/',
            force_update=False,
            use_tfrecord=False,  # TODO
            UPPER_BOUND=500.0,
            DOWNSAMPLE=2,
            WINSIZE=1000,
            NONOVERLAP=500,
            NONOVERLAP_SMALL_TTF=10,
            TRAIN_RATIO=0.7):
        """ 
        - [ ] categorical feature -> should be embedding or one-hot format.
        - [v] remove NaN
        - [v] normalize input/output
        - [v] validate intervals between records are equivalent.
        - [v] windowing: a question to ask: How long is sensor signals used to judge abnormal by domain experts? (WINSIZE, DOWNSAMPLE, NONOVERLAP)
        - [ ] handle data imbalanced: augment data which close to failures by randomly downsampling mechanism.
        - [v] handle data imbalanced: make sure within one batch, number of normal data is equivalent to number of abnormal data.
        - [v] setup RUL upper bound, because there is no valuable information within the large ttf value data, a question to ask: what is the longest time between abnormal signal and machine failure in domain expert opinions? (UPPER_BOUND)
        - [ ] preprocess training data to TFRecords for high performance pipeline (optional for small data)
        - [ ] feature selection by domain expert, visualization might help. xgboost, LIME, shap might be useful to interpret the model.
        
        preprocessing:
            x: DOWNSAMPLE -> WINSIZE -> NONOVERLAP
            y: clip(0, UPPER_BOUND)
        
        Optional to save records into TFReocrds, not very large actually.
        """
        super(PHM2018, self).__init__()
        self.root_path = root_path
        self.numpy_path = os.path.join(
            self.root_path,
            'u{}d{}w{}n{}t{}'.format(UPPER_BOUND, DOWNSAMPLE, WINSIZE,
                                     NONOVERLAP, TRAIN_RATIO))
        if not os.path.exists(self.numpy_path):
            os.makedirs(self.numpy_path)
        self.force_update = force_update
        self.use_tfrecord = use_tfrecord
        self.UPPER_BOUND = UPPER_BOUND
        self.DOWNSAMPLE = DOWNSAMPLE
        self.WINSIZE = WINSIZE
        self.NONOVERLAP = NONOVERLAP
        self.NONOVERLAP_SMALL_TTF = NONOVERLAP_SMALL_TTF
        self.TRAIN_RATIO = TRAIN_RATIO

        self.data = self._get_tfdset()

    def get_data(self):
        return self.data

    def _get_tfdset(self):
        all_big_ttf_x_path = os.path.join(self.numpy_path, 'all_big_ttf_x.npy')
        all_big_ttf_y_path = os.path.join(self.numpy_path, 'all_big_ttf_y.npy')
        all_small_ttf_x_path = os.path.join(self.numpy_path,
                                            'all_small_ttf_x.npy')
        all_small_ttf_y_path = os.path.join(self.numpy_path,
                                            'all_small_ttf_y.npy')

        if os.path.exists(all_big_ttf_x_path) and not self.force_update:
            all_big_ttf_x_np = np.load(all_big_ttf_x_path)
            all_big_ttf_y_np = np.load(all_big_ttf_y_path)
            all_small_ttf_x_np = np.load(all_small_ttf_x_path)
            all_small_ttf_y_np = np.load(all_small_ttf_y_path)
        else:
            all_big_ttf_x_np = []
            all_big_ttf_y_np = []
            all_small_ttf_x_np = []
            all_small_ttf_y_np = []
            for data_path in tqdm(
                    glob.glob(os.path.join(self.root_path, 'train/*.csv'))):
                LOGGER.info('processing {} ...'.format(data_path))
                file_name = os.path.basename(data_path)
                ttf_path = os.path.join(self.root_path, 'train/train_ttf',
                                        file_name)
                (big_ttf_x_np,
                 big_ttf_y_np), (small_ttf_x_np,
                                 small_ttf_y_np) = self._parse_one_csv(
                                     data_path, ttf_path)
                if len(big_ttf_x_np) > 0:
                    all_big_ttf_x_np.append(big_ttf_x_np)
                    all_big_ttf_y_np.append(big_ttf_y_np)
                if len(small_ttf_x_np) > 0:
                    all_small_ttf_x_np.append(small_ttf_x_np)
                    all_small_ttf_y_np.append(small_ttf_y_np)
            all_big_ttf_x_np = np.concatenate(all_big_ttf_x_np, axis=0)
            all_big_ttf_y_np = np.concatenate(all_big_ttf_y_np, axis=0)
            all_small_ttf_x_np = np.concatenate(all_small_ttf_x_np, axis=0)
            all_small_ttf_y_np = np.concatenate(all_small_ttf_y_np, axis=0)

            np.save(all_big_ttf_x_path, all_big_ttf_x_np)
            np.save(all_big_ttf_y_path, all_big_ttf_y_np)
            np.save(all_small_ttf_x_path, all_small_ttf_x_np)
            np.save(all_small_ttf_y_path, all_small_ttf_y_np)

        big_num_train = int(len(all_big_ttf_y_np) * self.TRAIN_RATIO)
        small_num_train = int(len(all_small_ttf_y_np) * self.TRAIN_RATIO)

        train_np_x_big_ttf, valid_np_x_big_ttf = np.split(
            all_big_ttf_x_np, [
                big_num_train,
            ])
        train_np_y_big_ttf, valid_np_y_big_ttf = np.split(
            all_big_ttf_y_np, [
                big_num_train,
            ])

        train_np_x_small_ttf, valid_np_x_small_ttf = np.split(
            all_small_ttf_x_np, [
                small_num_train,
            ])
        train_np_y_small_ttf, valid_np_y_small_ttf = np.split(
            all_small_ttf_y_np, [
                small_num_train,
            ])

        ## normalization ## on train_np_y_small_ttf
        mean = np.mean(train_np_x_small_ttf, axis=0)
        std = np.std(train_np_x_small_ttf, axis=0)
        train_np_x_big_ttf = (train_np_x_big_ttf - mean) / std
        train_np_y_big_ttf = train_np_y_big_ttf / self.UPPER_BOUND
        train_np_x_small_ttf = (train_np_x_small_ttf - mean) / std
        train_np_y_small_ttf = train_np_y_small_ttf / self.UPPER_BOUND

        valid_np_x_all = np.concatenate(
            [valid_np_x_big_ttf, valid_np_x_small_ttf], axis=0)
        valid_np_y_all = np.concatenate(
            [valid_np_y_big_ttf, valid_np_y_small_ttf], axis=0)
        valid_np_x_all = (valid_np_x_all - mean) / std
        valid_np_y_all = valid_np_y_all / self.UPPER_BOUND

        LOGGER.info(train_np_x_big_ttf.shape)
        LOGGER.info(train_np_y_big_ttf.shape)
        LOGGER.info(train_np_x_small_ttf.shape)
        LOGGER.info(train_np_y_small_ttf.shape)
        LOGGER.info(valid_np_x_all.shape)
        LOGGER.info(valid_np_y_all.shape)
        LOGGER.info('X mean\n{}\nstd:\n{}'.format(mean, std))

        # Train
        train_big_dset = tf.data.Dataset.from_tensor_slices(
            (train_np_x_big_ttf, train_np_y_big_ttf))
        train_big_dset = train_big_dset.cache().shuffle(100000).batch(
            FLAGS.bs // 2, drop_remainder=True)
        train_small_dset = tf.data.Dataset.from_tensor_slices(
            (train_np_x_small_ttf, train_np_y_small_ttf))
        train_small_dset = train_small_dset.cache().shuffle(100000).batch(
            FLAGS.bs // 2, drop_remainder=True)
        train_dset = tf.data.Dataset.zip((train_big_dset, train_small_dset))
        train_dset = train_dset.flat_map(
            lambda big_ttf, small_ttf: tf.data.Dataset.from_tensors(big_ttf).
            concatenate(tf.data.Dataset.from_tensors(small_ttf))).prefetch(
                buffer_size=tf.data.experimental.AUTOTUNE)
        # Validation
        valid_dset = tf.data.Dataset.from_tensor_slices(
            (valid_np_x_all, valid_np_y_all))
        valid_dset = valid_dset.cache().batch(
            FLAGS.bs, drop_remainder=False).prefetch(
                buffer_size=tf.data.experimental.AUTOTUNE)

        return train_dset, valid_dset

    def _parse_one_csv(self, data_path, ttf_path):
        # read x and y dataframe from csv
        df_data = pd.read_csv(data_path)
        df_ttf = pd.read_csv(ttf_path)
        all_df = pd.merge(df_data, df_ttf, how='inner', on='time')
        ## setup RUL upper bound ##
        all_df.iloc[:, -3:] = all_df.iloc[:, -3:].fillna(self.UPPER_BOUND)
        all_df.iloc[:, -3:] = all_df.iloc[:, -3:].clip(0, self.UPPER_BOUND)
        ## remove NaN ##
        all_df.dropna(axis=0, inplace=True)
        ROW_INTERVAL = all_df.time.diff().mode().values[0]
        # split into train and validation
        total_len = all_df.values.shape[0]
        all_df.drop(columns='Tool', inplace=True)  # TODO !!!
        all_df = all_df.astype('float32')
        big_ttf_df = all_df[(all_df.iloc[:, -3:] >= self.UPPER_BOUND).any(
            axis=1)]
        small_ttf_df = all_df[(all_df.iloc[:, -3:] < self.UPPER_BOUND).any(
            axis=1)]

        def wrap_as_np(df, nonoverlap):
            ## validate intervals between records are equivalent. ##
            segment_idx_df = df[(df.time.diff() != ROW_INTERVAL)]
            segment_idx = segment_idx_df.index.values
            ## windowing ## WINSIZE, DOWNSAMPLE, NONOVERLAP
            start = 0
            result_x = []
            result_y = []
            data_x = df.values[::self.DOWNSAMPLE, :-3]
            data_y = df.values[::self.DOWNSAMPLE, -3:]
            for end in tqdm(segment_idx[1:]):
                tmp_x = data_x[start:end]
                tmp_y = data_y[start:end]
                if tmp_x.shape[0] > self.WINSIZE:  # big enough
                    tmp_result_x = []
                    # -1 because x(t-k)~x(t) -> y(t)
                    for i in range(0, tmp_x.shape[0] - self.WINSIZE + 1,
                                   nonoverlap):
                        tmp_result_x.append(tmp_x[i:i + self.WINSIZE])
                    tmp_result_x = np.array(tmp_result_x)
                    tmp_result_y = tmp_y[self.WINSIZE - 1::nonoverlap]
                    result_x.append(tmp_result_x)
                    result_y.append(tmp_result_y)
                start = end
            if len(result_x) > 0:
                result_x = np.concatenate(result_x, axis=0)
                result_y = np.concatenate(result_y, axis=0)
                assert result_x.shape[0] == result_y.shape[0]
            return result_x, result_y

        big_ttf_x_np, big_ttf_y_np = wrap_as_np(big_ttf_df, self.NONOVERLAP)
        small_ttf_x_np, small_ttf_y_np = wrap_as_np(small_ttf_df,
                                                    self.NONOVERLAP_SMALL_TTF)
        return (big_ttf_x_np, big_ttf_y_np), (small_ttf_x_np, small_ttf_y_np)


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

# split train into big ttf and small ttf numpy, for balancing batch when training
#             def get_small_ttf_mask(nparray):
#                 indicies = []
#                 for index, ttfs in enumerate(nparray):
#                     for ttf in ttfs:
#                         if ttf < self.UPPER_BOUND:
#                             indicies.append(index)
#                             break
#                 mask = np.zeros([nparray.shape[0],],dtype=bool)
#                 mask[indicies]=True
#                 return mask

#             mask = get_small_ttf_mask(train_np_y_big_ttf)
#             train_np_x_small_ttf = train_np_x_big_ttf[mask]
#             train_np_y_small_ttf = train_np_y_big_ttf[mask]
#             train_np_x_big_ttf = train_np_x_big_ttf[~mask]
#             train_np_y_big_ttf = train_np_y_big_ttf[~mask]
