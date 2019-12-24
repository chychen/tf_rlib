import os
import time
import numpy as np
import copy
from tqdm.auto import tqdm
from absl import flags, logging
import tensorflow as tf
from tensorflow.python.eager import profiler
from tf_rlib.runners import MetricsManager
from tf_rlib import utils, metrics

FLAGS = flags.FLAGS
LOGGER = logging.get_absl_logger()


class Runner:
    """ please make sure all the losses follow the distributed training mechanism:
    please see https://www.tensorflow.org/tutorials/distribute/custom_training
    
    # required implemented functions/properties are:
    - init()
    - train_step()
    - validate_step() 
    - required_flags()
    
    """
    def __init__(self, train_dataset, valid_dataset=None, best_state=None):
        """
        Args
            models (dict): key(str), value(tf.keras.)
            metrics (dict): key(str), value(tf.keras.metrics.Metric)
        """
        FLAGS.exp_name = self.__class__.__name__
        self._validate_required_flags()

        utils.init_tf_rlib(show=True)
        self.strategy = tf.distribute.MirroredStrategy()
        LOGGER.info('Number of devices: {}'.format(
            self.strategy.num_replicas_in_sync))

        self.epoch = 0
        self.step = 0
        self.save_path = FLAGS.save_path
        self.best_state = best_state
        self.metrics_manager = MetricsManager(best_state)
        self.models_outs_shapes = dict()

        with self._get_strategy_ctx():
            self.models, train_metrics, valid_metrics = self.init()
            if self.strategy.num_replicas_in_sync >= 1:  #TODO: BUG!!!!
                self.train_dataset = self.strategy.experimental_distribute_dataset(
                    train_dataset)
                self.valid_dataset = self.strategy.experimental_distribute_dataset(
                    valid_dataset)

            # weights init in first call()
            for key, model in self.models.items():
                model_outs = model(next(iter(self.train_dataset))[0])
                if type(model_outs) != tuple:
                    model_outs = tuple((model_outs, ))
                outs_shapes = tuple((out.shape for out in model_outs))
                self.models_outs_shapes[key] = outs_shapes
                LOGGER.info('{} model contains {} trainable variables.'.format(
                    key, model.num_params))
                model.summary(print_fn=LOGGER.info)
            if train_metrics is None or valid_metrics is None:
                raise ValueError(
                    'metrics are required, Note: please use tf.keras.metrics.MeanTensor to compute the training loss, which is more efficient by avoiding redundant tain loss computing.'
                )
            else:
                for k, v in train_metrics.items():
                    self.metrics_manager.add_metrics(k, v,
                                                     MetricsManager.KEY_TRAIN)
                for k, v in valid_metrics.items():
                    self.metrics_manager.add_metrics(k, v,
                                                     MetricsManager.KEY_VALID)
        if valid_metrics is not None:
            for k, v in valid_metrics.items():
                if v.__class__.__name__ in metrics.__dict__:
                    self.metrics_manager.add_metrics(
                        k, metrics.__dict__[v.__class__.__name__](
                            MetricsManager.KEY_TEST), MetricsManager.KEY_TEST)
                else:
                    self.metrics_manager.add_metrics(
                        k, tf.keras.metrics.__dict__[v.__class__.__name__](
                            MetricsManager.KEY_TEST), MetricsManager.KEY_TEST)

    def init(self):
        raise NotImplementedError

    def train_step(self, x, y):
        """ NOTE: drop_remainder=False in dataset will introduce bias in training 
        Args:
            x (batch): mini batch data
            y (batch): mini batch label
        Returns:
            losses (dict)
        """
        raise NotImplementedError

    @tf.function
    def _train_step(self, x, y):
        def train_fn(x, y):
            metrics = self.train_step(x, y)
            self.metrics_manager.update(metrics, MetricsManager.KEY_TRAIN)

        if self.strategy.num_replicas_in_sync > 1:
            self.strategy.experimental_run_v2(train_fn, args=(x, y))
        else:
            train_fn(x, y)

    @tf.function
    def test(self, x, y):
        self.validate_step(x, y)

    def validate_step(self, x, y):
        """
        Args:
            x (batch): mini batch data
            y (batch): mini batch label
        Returns:
            metrics (dict)
        """
        raise NotImplementedError

    @tf.function
    def _validate_step(self, x, y):
        def valid_fn(x, y):
            metrics = self.validate_step(x, y)
            self.metrics_manager.update(metrics, MetricsManager.KEY_VALID)

        if self.strategy.num_replicas_in_sync > 1:
            self.strategy.experimental_run_v2(valid_fn, args=(x, y))
        else:
            valid_fn(x, y)

    @tf.function
    def inference(self, dataset):
        raise NotImplementedError

    @property
    def required_flags(self):
        """ a list, template runner will validate required_flags defined here such as FLAGS.bs, FLAGS.dim ... etc.
        """
        raise NotImplementedError

    def _validate_required_flags(self):
        if self.required_flags is not None:
            for key in self.required_flags:
                if FLAGS.flag_values_dict()[key] is None:
                    raise ValueError('FLAGS.{} should not be None'.format(key))

    def evaluate(self, dataset=None):
        """ inference on single gpu only
        """
        if dataset is None:
            dataset = self.valid_dataset
        for _, (x_batch, y_batch) in enumerate(dataset):
            metrics = self.validate_step(x_batch, y_batch)
            self.metrics_manager.update(metrics, MetricsManager.KEY_TEST)

        return self.metrics_manager.get_result(keys=[MetricsManager.KEY_TEST])

    def begin_fit_callback(self, lr):
        pass

    def begin_epoch_callback(self, epoch_id, epochs):
        pass

    def fit(self, epochs, lr):
        with self._get_strategy_ctx():
            self.begin_fit_callback(lr)
            train_pbar = tqdm(desc='train', leave=False, dynamic_ncols=True)
            valid_pbar = tqdm(desc='valid', leave=False, dynamic_ncols=True)
            for e_idx in range(epochs):
                train_num_batch = 0
                valid_num_batch = 0
                first_e_timer = time.time()
                self.begin_epoch_callback(self.epoch, epochs)
                self.epoch = self.epoch + 1
                # progress bars
                train_pbar.reset()
                valid_pbar.reset()
                self.metrics_manager.reset()
                # train one epoch
                for train_num_batch, (x_batch, y_batch) in enumerate(
                        self.train_dataset):
                    self.step = self.step + 1
                    # train one step
                    if FLAGS.profile:
                        with profiler.Profiler(
                                os.path.join(FLAGS.log_path, 'profile')):
                            self._train_step(x_batch, y_batch)
                    else:
                        self._train_step(x_batch, y_batch)
                    train_pbar.update(1)
                    train_pbar.set_postfix(
                        {'epoch/total': '{}/{}'.format(self.epoch, epochs)})
                self._log_data(x_batch, y_batch, training=True)

                # validate one epoch
                if self.valid_dataset is not None:
                    for valid_num_batch, (x_batch, y_batch) in enumerate(
                            self.valid_dataset):
                        # validate one step
                        self._validate_step(x_batch, y_batch)
                        valid_pbar.update(1)
                    if self.metrics_manager.is_better_state():
                        self.save_best()
                        valid_pbar.set_postfix({
                            'best epoch':
                            self.epoch,
                            self.best_state:
                            self.metrics_manager.best_record
                        })
                    self._log_data(x_batch, y_batch, training=False)

                if self.epoch == 1:
                    LOGGER.warn('')  # new line
                    LOGGER.warn('Time cost for first epoch: {} sec'.format(
                        time.time() - first_e_timer))
                if e_idx == 0:
                    train_num_batch = train_num_batch + 1
                    valid_num_batch = valid_num_batch + 1
                    self.metrics_manager.set_num_batch(train_num_batch,
                                                       valid_num_batch)
                    train_pbar.close()
                    valid_pbar.close()
                    train_pbar = tqdm(desc='train',
                                      leave=False,
                                      dynamic_ncols=True,
                                      total=train_num_batch)
                    valid_pbar = tqdm(desc='valid',
                                      leave=False,
                                      dynamic_ncols=True,
                                      total=valid_num_batch)

                # logging
                self.metrics_manager.show_message(self.epoch)
            self.metrics_manager.register_hparams()

    def save(self, path):
        for key, model in self.models.items():
            model.save_weights(os.path.join(path, key))

    def load(self, path):
        for key, model in self.models.items():
            model.load_weights(os.path.join(path, key))

    def save_best(self):
        self.save(os.path.join(self.save_path, 'best'))

    def load_best(self):
        self.load(os.path.join(self.save_path, 'best'))

    def log_scalar(self, name, value, step, training):
        key = MetricsManager.KEY_TRAIN if training else MetricsManager.KEY_VALID
        self.metrics_manager.add_scalar(name,
                                        value,
                                        step,
                                        key,
                                        tag=MetricsManager.TAG_HPARAMS)

    @property
    def best_state_record(self):
        return self.metrics_manager.best_record

    def _get_size(self, dataset):
        num_batch = 0
        num_data = 0
        if dataset is not None:
            for data in dataset:
                num_batch += 1
                num_data += data.shape[0]
        return num_batch, num_data

    def _get_strategy_ctx(self):
        if self.strategy.num_replicas_in_sync > 1:
            strategy_context = self.strategy.scope()
        else:
            strategy_context = utils.dummy_context_mgr()
        return strategy_context

    def _log_data(self, x_batch, y_batch, training):
        key = MetricsManager.KEY_TRAIN if training else MetricsManager.KEY_VALID
        # log images
        # vis x, y if images
        names = ['x', 'y']
        batches = [x_batch, y_batch]
        # vis model outputs if they are images!
        for model_key, model in self.models.items():
            # sometime model output more than one result.
            out_shapes = self.models_outs_shapes[model_key]
            valid_out_idx = []
            for i, shape in enumerate(out_shapes):
                if len(shape) == 4 and shape[-1] <= 4:
                    valid_out_idx.append(i)
            # only inference model when necessary
            if len(valid_out_idx) > 0:
                model_out = model(x_batch)
                for idx in valid_out_idx:
                    names.append(model_key + '_' + str(idx))
                    batches.append(model_out[idx])
        # vis
        for name, batch in zip(names, batches):
            if self.strategy.num_replicas_in_sync == 1:
                batch_local = batch
            else:
                batch_local = batch.values[0]

            if type(batch_local) == tuple:
                for i, sub_batch in enumerate(batch_local):
                    # others
                    if len(
                            sub_batch.shape
                    ) == 4 and sub_batch.shape[-1] <= 4:  # [b, w, h, c], c<4
                        self.metrics_manager.show_image(sub_batch,
                                                        key,
                                                        epoch=self.epoch,
                                                        name=name +
                                                        '_tuple_{}'.format(i))
                    # few-shot
                    few_shot_name = ['support', 'query']
                    if len(sub_batch.shape) == 7 and sub_batch.shape[
                            -1] <= 4:  # [b, q, c, k, w, h, c], c<4
                        sub_batch = tf.reshape(sub_batch,
                                               [-1, *sub_batch.shape[-3:]])
                        self.metrics_manager.show_image(sub_batch,
                                                        key,
                                                        epoch=self.epoch,
                                                        name=name + '_' +
                                                        few_shot_name[i])
            else:
                if len(
                        batch_local.shape
                ) == 4 and batch_local.shape[-1] <= 4:  # [b, w, h, c], c<4
                    self.metrics_manager.show_image(batch_local,
                                                    key,
                                                    epoch=self.epoch,
                                                    name=name)
