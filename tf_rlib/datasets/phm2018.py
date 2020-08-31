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
            ROW_INTERVAL=4,  # = all_df.time.diff().mode().values[0]
            DOWNSAMPLE=2,  # ratio
            UPPER_BOUND=1000,  # seconds
            NONOVERLAP=500,  # seconds, stirde=int(NONOVERLAP/ROW_INTERVAL/DOWNSAMPLE)
            NONOVERLAP_SMALL_TTF=25,  # seconds, stirde=int(NONOVERLAP_SMALL_TTF/ROW_INTERVAL/DOWNSAMPLE)
            TRAIN_RATIO=0.7,
            TRAIN_DATA_PER_EPOCH=100000):
        """ one failure could generate UPPER_BOUND/(NONOVERLAP_SMALL_TTF*DOWNSAMPLE*ROW_INTERVAL(4s)) amount for training/validating abnormal data
        UPPER_BOUND: equal to window size for each data, also used to seperate normal and abnormal!!
        
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
        self.force_update = force_update
        self.use_tfrecord = use_tfrecord
        self.ROW_INTERVAL = ROW_INTERVAL
        self.UPPER_BOUND = UPPER_BOUND
        self.DOWNSAMPLE = DOWNSAMPLE
        self.WINSIZE = int(UPPER_BOUND / ROW_INTERVAL / DOWNSAMPLE)
        self.NONOVERLAP = NONOVERLAP
        self.NONOVERLAP_STRIDE = int(NONOVERLAP / ROW_INTERVAL / DOWNSAMPLE)
        self.NONOVERLAP_SMALL_TTF = NONOVERLAP_SMALL_TTF
        self.NONOVERLAP_SMALL_TTF_STRIDE = int(NONOVERLAP_SMALL_TTF /
                                               ROW_INTERVAL / DOWNSAMPLE)
        self.TRAIN_RATIO = TRAIN_RATIO
        self.TRAIN_DATA_PER_EPOCH = TRAIN_DATA_PER_EPOCH
        self.numpy_path = os.path.join(
            self.root_path,
            'UPPERBOUND{}DownSample{}WSIZE{}NOverNormal{}NOverAB{}TrainRatio{}'
            .format(self.UPPER_BOUND, self.DOWNSAMPLE, self.WINSIZE,
                    self.NONOVERLAP, self.NONOVERLAP_SMALL_TTF,
                    self.TRAIN_RATIO))
        LOGGER.info('data locate at: {}'.format(self.numpy_path))
        if not os.path.exists(self.numpy_path):
            os.makedirs(self.numpy_path)

    def get_data(self):
        return self._get_tfdset()

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

        # split into train and validation
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

        LOGGER.info('train normal data shape:{}'.format(
            train_np_x_big_ttf.shape))
        LOGGER.info('train normal label shape:{}'.format(
            train_np_y_big_ttf.shape))
        LOGGER.info('train abnormal data shape:{}'.format(
            train_np_x_small_ttf.shape))
        LOGGER.info('train abnormal label shape:{}'.format(
            train_np_y_small_ttf.shape))
        LOGGER.info('valid shape:{}'.format(valid_np_x_all.shape))
        LOGGER.info('label shape:{}'.format(valid_np_y_all.shape))
        LOGGER.info('X mean\n{}\nstd:\n{}'.format(mean, std))

        # Train
        train_big_dset = tf.data.Dataset.from_tensor_slices(
            (train_np_x_big_ttf, train_np_y_big_ttf))
        train_big_dset = train_big_dset.cache().shuffle(100000).repeat().take(
            self.TRAIN_DATA_PER_EPOCH // 2)
        train_small_dset = tf.data.Dataset.from_tensor_slices(
            (train_np_x_small_ttf, train_np_y_small_ttf))
        train_small_dset = train_small_dset.cache().shuffle(
            100000).repeat().take(self.TRAIN_DATA_PER_EPOCH // 2)
        train_dset = tf.data.Dataset.zip((train_big_dset, train_small_dset))

        def merge(big_ttf, small_ttf):
            dset = tf.data.Dataset.from_tensors(big_ttf)
            dset = dset.concatenate(tf.data.Dataset.from_tensors(small_ttf))
            return dset

        train_dset = train_dset.flat_map(merge).batch(
            FLAGS.bs, drop_remainder=True).prefetch(
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
        all_df.drop(columns='Tool', inplace=True)  # TODO !!!
        all_df = all_df.astype('float32')
        big_ttf_df = all_df[(all_df.iloc[:, -3:] >= self.UPPER_BOUND).any(
            axis=1)]
        small_ttf_df = all_df[(all_df.iloc[:, -3:] < self.UPPER_BOUND).any(
            axis=1)]

        #         ## validate intervals between records are equivalent. ##
        #         def wrap_as_np(df, nonoverlap):
        #             df = df.reset_index()
        #             segment_idx_df = df[(df.time.diff() != self.ROW_INTERVAL)]
        #             segment_idx = segment_idx_df.index.values
        #             ## windowing ## WINSIZE, DOWNSAMPLE, NONOVERLAP
        #             start = 0
        #             result_x = []
        #             result_y = []
        #             data_x = df.values[::self.DOWNSAMPLE, :-3]
        #             data_y = df.values[::self.DOWNSAMPLE, -3:]
        #             for end in tqdm(segment_idx[1:]):
        #                 tmp_x = data_x[start:end]
        #                 tmp_y = data_y[start:end]
        #                 if tmp_x.shape[0] > self.WINSIZE:  # big enough
        #                     tmp_result_x = []
        #                     # -1 because x(t-k)~x(t) -> y(t)
        #                     for i in range(0, tmp_x.shape[0] - self.WINSIZE + 1,
        #                                    nonoverlap):
        #                         tmp_result_x.append(tmp_x[i:i + self.WINSIZE])
        #                     tmp_result_x = np.array(tmp_result_x)
        #                     tmp_result_y = tmp_y[self.WINSIZE - 1::nonoverlap]
        #                     result_x.append(tmp_result_x)
        #                     result_y.append(tmp_result_y)
        #                 start = end
        #             if len(result_x) > 0:
        #                 result_x = np.concatenate(result_x, axis=0)
        #                 result_y = np.concatenate(result_y, axis=0)
        #                 assert result_x.shape[0] == result_y.shape[0]
        #             return result_x, result_y

        def wrap_as_np(df, nonoverlap):
            result_x = []
            result_y = []
            data_x = df.values[::self.DOWNSAMPLE, :-3]
            data_y = df.values[::self.DOWNSAMPLE, -3:]
            if data_x.shape[0] > self.WINSIZE:  # big enough
                tmp_result_x = []
                # -1 because x(t-k)~x(t) -> y(t)
                for i in range(0, data_x.shape[0] - self.WINSIZE + 1,
                               nonoverlap):
                    tmp_result_x.append(data_x[i:i + self.WINSIZE])
                tmp_result_x = np.array(tmp_result_x)
                tmp_result_y = data_y[self.WINSIZE - 1::nonoverlap]
                result_x.append(tmp_result_x)
                result_y.append(tmp_result_y)
            if len(result_x) > 0:
                result_x = np.concatenate(result_x, axis=0)
                result_y = np.concatenate(result_y, axis=0)
                assert result_x.shape[0] == result_y.shape[0]
            return result_x, result_y

        big_ttf_x_np, big_ttf_y_np = wrap_as_np(big_ttf_df,
                                                self.NONOVERLAP_STRIDE)
        small_ttf_x_np, small_ttf_y_np = wrap_as_np(
            small_ttf_df, self.NONOVERLAP_SMALL_TTF_STRIDE)
        return (big_ttf_x_np, big_ttf_y_np), (small_ttf_x_np, small_ttf_y_np)
