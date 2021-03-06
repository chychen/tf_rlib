import os
import sys
import traceback
import time
import numpy as np
import copy
from tqdm.auto import tqdm
from absl import flags, logging
import tensorflow as tf
from tensorflow.python.eager import profiler
from tf_rlib.runners.base.metrics_manager import MetricsManager
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
    - validation_loop() # optional
    
    # optional features: default False, please overwrite it to return True if the inheritted runner supports the features.
    - support_amp()
    
    """
    def __init__(self, train_dataset, valid_dataset=None, best_state=None):
        """
        Args
            models (dict): key(str), value(tf.keras.)
            metrics (dict): key(str), value(tf.keras.metrics.Metric)
        """
        if FLAGS.exp_name == 'default':
            FLAGS.exp_name = self.__class__.__name__
        self._validate_flags()

        utils.init_tf_rlib(show=True)
        self.strategy = tf.distribute.MirroredStrategy()
        LOGGER.info('Number of devices: {}'.format(
            self.strategy.num_replicas_in_sync))

        self.global_epoch = 0
        self.global_step = 0
        self.train_num_batch = 0
        self.valid_num_batch = 0
        self.save_path = FLAGS.save_path
        self.best_epoch = 0
        self.best_state = best_state
        self.metrics_manager = MetricsManager(best_state)
        self.models_outs_shapes = dict()

        with self._get_strategy_ctx():
            self.models, self.models_inputs_shape, train_metrics, valid_metrics = self.init(
            )
            if self.strategy.num_replicas_in_sync > 1:  #TODO: BUG!!!!
                self.train_dataset = self.strategy.experimental_distribute_dataset(
                    train_dataset)
                if valid_dataset is not None:
                    self.valid_dataset = self.strategy.experimental_distribute_dataset(
                        valid_dataset)
            else:
                self.train_dataset = train_dataset
                if valid_dataset is not None:
                    self.valid_dataset = valid_dataset

            # if shape isn't specified, use shape in dataset
            if self.models_inputs_shape is None:
                self.models_inputs_shape = {}
                for key, model in self.models.items():
                    if type(self.train_dataset.element_spec[0]) != tuple:
                        self.models_inputs_shape[key] = [
                            train_dataset.element_spec[0].shape[1:]
                        ]
                    else:
                        self.models_inputs_shape[key] = [
                            x.shape[1:] for x in train_dataset.element_spec[0]
                        ]
            else:
                for key in self.models_inputs_shape:
                    self.models_inputs_shape[key] = [
                        self.models_inputs_shape[key]
                    ]
            # weights init in first call()
            inputs, outputs = [], []
            for key, model in self.models.items():
                keras_input = tuple()
                for shape in self.models_inputs_shape[key]:
                    keras_input = keras_input + (tf.keras.Input(shape), )
                if len(keras_input) == 1:
                    keras_input = keras_input[0]
                model_outs = model(keras_input, training=False)
                inputs.append(keras_input)
                outputs.append(model_outs)
                if type(model_outs) != tuple:
                    model_outs = tuple((model_outs, ))
                outs_shapes = tuple((out.shape for out in model_outs))
                self.models_outs_shapes[key] = outs_shapes
                LOGGER.info('model: {}, input_shape: {}'.format(
                    key, self.models_inputs_shape[key]))
                model.summary(print_fn=LOGGER.info)
            self.model_inputs = inputs
            self.model_outputs = outputs
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
                try:
                    if v.__class__.__name__ in metrics.__dict__:
                        self.metrics_manager.add_metrics(
                            k, metrics.__dict__[v.__class__.__name__](
                                name=v.__class__.__name__ + '_' +
                                MetricsManager.KEY_TEST),
                            MetricsManager.KEY_TEST)
                    else:
                        self.metrics_manager.add_metrics(
                            k, tf.keras.metrics.__dict__[v.__class__.__name__](
                                name=v.__class__.__name__ + '_' +
                                MetricsManager.KEY_TEST),
                            MetricsManager.KEY_TEST)
                except:
                    LOGGER.warn(
                        '{} metric is not valid for testing mode'.format(k))

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
            self.strategy.run(train_fn, args=(x, y))
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
            self.strategy.run(valid_fn, args=(x, y))
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

    @property
    def support_amp(self):
        """ boolean, default False, please claim your runner with True when you support the AMP feature.
        """
        return False

    def _validate_flags(self):
        # validate required_flags
        if self.required_flags is not None:
            for key in self.required_flags:
                if FLAGS.flag_values_dict()[key] is None:
                    raise ValueError('FLAGS.{} should not be None'.format(key))
        # validate features
        if FLAGS.amp and not self.support_amp:
            raise ValueError(
                'FLAGS.amp should not be True, the running runner haven\'t supported amp yet, or the runner didn\'t define class property support_amp=True (default=False)'
                .format(key))

    def evaluate(self, dataset=None):
        """ inference on single gpu only
        """
        if dataset is None:
            dataset = self.valid_dataset
        for _, (x_batch, y_batch) in enumerate(dataset):
            metrics = self.validate_step(x_batch, y_batch)
            self.metrics_manager.update(metrics, MetricsManager.KEY_TEST)
        res = self.metrics_manager.get_result(keys=[MetricsManager.KEY_TEST])
        self.metrics_manager.reset_metrics(key=MetricsManager.KEY_TEST)
        return res

    def begin_fit_callback(self, lr):
        pass

    def begin_epoch_callback(self, epoch_id, epochs):
        pass

    def _begin_epoch_callback(self, epoch_id, epochs):
        if FLAGS.amp:
            self.log_scalar('loss_scale', self.optim.loss_scale(), True)
        self.begin_epoch_callback(epoch_id, epochs)

    def begin_step_callback(self, step_id, epochs):
        pass

    def fit(self, epochs, lr, find_best=False):
        LOGGER.debug('starting fit')
        global_total_epochs = self.global_epoch + epochs
        with self._get_strategy_ctx() as ctx:
            self.begin_fit_callback(lr)
            try:
                train_pbar = tqdm(desc='train',
                                  leave=False,
                                  dynamic_ncols=True,
                                  disable=not FLAGS.tqdm)
                valid_pbar = tqdm(desc='valid',
                                  leave=False,
                                  dynamic_ncols=True,
                                  disable=not FLAGS.tqdm)
                epoch_stride = FLAGS.pre_augment if FLAGS.pre_augment is not None else 1
                step_idx = 0
                for e_idx in range(0, epochs, epoch_stride):
                    is_last_run = (e_idx //
                                   epoch_stride) == (epochs // epoch_stride -
                                                     1)
                    self.train_num_batch = 0
                    self.valid_num_batch = 0
                    first_e_timer = time.time()
                    self._begin_epoch_callback(e_idx+1, epochs)
                    self.global_epoch = self.global_epoch + 1 * epoch_stride
                    # progress bars
                    if FLAGS.tqdm:
                        train_pbar.reset()
                        valid_pbar.reset()
                    self.metrics_manager.reset()

                    # train one epoch
                    if is_last_run and find_best:
                        try:
                            self.load_best()
                        except:
                            LOGGER.info('Best model not exists.')
                    train_pbar.set_postfix({
                        'epoch/total':
                        '{}/{}'.format(self.global_epoch, global_total_epochs)
                    })
                    LOGGER.debug('starting training epoch')
                    for _, (x_batch, y_batch) in enumerate(self.train_dataset):
                        self.global_step = self.global_step + 1
                        step_idx = step_idx + 1
                        self.begin_step_callback(step_idx, epochs)
                        if FLAGS.profile:
                            with profiler.Profiler(
                                    os.path.join(FLAGS.log_path, 'profile')):
                                self._train_step(x_batch, y_batch)
                        else:
                            self._train_step(x_batch, y_batch)
                        train_pbar.update(1)
                        self.train_num_batch = self.train_num_batch + 1

                        # find best model by eavluating validation data on each training batch
                        if is_last_run and find_best:
                            if FLAGS.tqdm:
                                valid_pbar.reset()
                            self._validation_loop(valid_pbar)
                            # logging
                            self.metrics_manager.show_message(
                                self.global_epoch)
                            self.metrics_manager.reset()
                            self.global_epoch = self.global_epoch + 1
                    self._log_data(x_batch, y_batch, training=True)

                    LOGGER.debug('starting validaion epoch')
                    if not is_last_run or not find_best:
                        self._validation_loop(valid_pbar)

                    # others
                    if self.global_epoch == 1:
                        LOGGER.warn('')  # new line
                        LOGGER.warn('Time cost for first epoch: {} sec'.format(
                            time.time() - first_e_timer))
                    if e_idx == 0:
                        self.train_num_batch = self.train_num_batch + 1
                        self.valid_num_batch = self.valid_num_batch + 1
                        self.metrics_manager.set_num_batch(
                            self.train_num_batch, self.valid_num_batch)
                        if epochs > 1:
                            train_pbar.close()
                            valid_pbar.close()
                            train_pbar = tqdm(desc='train',
                                              leave=False,
                                              dynamic_ncols=True,
                                              total=self.train_num_batch,
                                              disable=not FLAGS.tqdm)
                            valid_pbar = tqdm(desc='valid',
                                              leave=False,
                                              dynamic_ncols=True,
                                              total=self.valid_num_batch,
                                              disable=not FLAGS.tqdm)
                    # logging
                    self.metrics_manager.show_message(self.global_epoch)
            except Exception as e:
                error_class = e.__class__.__name__
                detail = e.args[0]
                cl, exc, tb = sys.exc_info()
                lastCallStack = traceback.extract_tb(tb)[-1]
                fileName = lastCallStack[0]
                lineNum = lastCallStack[1]
                funcName = lastCallStack[2]
                LOGGER.error(
                    f"File \"{fileName}\", line {lineNum}, in {funcName}: [{error_class}] {detail}"
                )

            self.metrics_manager.register_hparams()

    def _validation_loop(self, valid_pbar):
        self.validation_loop(valid_pbar)

    def validation_loop(self, valid_pbar):
        if self.valid_dataset is not None:
            for valid_num_batch, (x_batch,
                                  y_batch) in enumerate(self.valid_dataset):
                LOGGER.debug('starting valid set')
                # validate one step
                self._validate_step(x_batch, y_batch)
                valid_pbar.update(1)
                self.valid_num_batch = self.valid_num_batch + 1
            if self.metrics_manager.is_better_state():
                self.save_best()
                self.best_epoch = self.global_epoch
            valid_pbar.set_postfix({
                'best epoch':
                self.best_epoch,
                self.best_state:
                self.metrics_manager.best_record
            })
            self._log_data(x_batch, y_batch, training=False)

    def save(self, path, key):
        self.models[key].save_weights(os.path.join(path, key))

    def get_saved_models_path(self):
        paths = {}
        for key in self.models:
            p = os.path.join(self.save_path, 'best', key)
            paths[key] = p
            print('{}: {}'.format(key, p))
        return paths

    def save_all(self, path):
        for key in self.models:
            self.save(path, key)

    def load(self, path, key):
        self.models[key].load_weights(path)

    def load_all(self, path):
        for key in self.models:
            self.load(os.path.join(path, key))

    def save_best(self):
        self.save_all(os.path.join(self.save_path, 'best'))

    def load_best(self):
        self.load_all(os.path.join(self.save_path, 'best'))

    def log_scalar(self, name, value, training, step=None):
        if step is None:
            step = self.global_epoch
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

    def custom_log_data(self, x_batch, y_batch):
        return None

    def _log_data(self, x_batch, y_batch, training):
        LOGGER.debug('getting first replica before logging')
        if self.strategy.num_replicas_in_sync > 1:
            x_batch = x_batch.values[0]
            y_batch = y_batch.values[0]
        key = MetricsManager.KEY_TRAIN if training else MetricsManager.KEY_VALID
        # log images
        # vis x, y if images
        names = []
        batches = []
        if hasattr(x_batch, 'shape'):
            names.append('x')
            batches.append(x_batch)
        if hasattr(y_batch, 'shape'):
            names.append('y')
            batches.append(y_batch)
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
                if x_batch.shape[1:] == model.input_shape[1:]:
                    model_out = model(x_batch, training=False)
                    for idx in valid_out_idx:
                        names.append(model_key + '_' + str(idx))
                        batches.append(model_out[idx])

        LOGGER.debug('starting custom_log_data')
        custom_dict = self.custom_log_data(x_batch, y_batch)
        if custom_dict is not None and type(custom_dict) == dict:
            for k, v in custom_dict.items():
                names.append(k)
                batches.append(v)

        LOGGER.debug('starting to vis log data')
        if len(batches) > 0:
            num_vis = 3 if batches[0].shape[0] > 3 else batches[0].shape[0]
            idx = np.random.choice(list(range(batches[0].shape[0])), num_vis)
            for name, batch in zip(names, batches):
                # randomly pick samples
                batch = tf.gather(batch, idx, axis=0)
                if type(batch) == tuple:
                    LOGGER.debug('tuple')
                    for i, sub_batch in enumerate(batch):
                        # others
                        if len(sub_batch.shape) == 4 and sub_batch.shape[
                                -1] <= 4:  # [b, w, h, c], c<4
                            self.metrics_manager.show_image(
                                sub_batch,
                                key,
                                epoch=self.global_epoch,
                                name=name + '_tuple_{}'.format(i))
                        # few-shot
                        few_shot_name = ['support', 'query']
                        if len(sub_batch.shape) == 7 and sub_batch.shape[
                                -1] <= 4:  # [b, q, c, k, w, h, c], c<4
                            sub_batch = tf.reshape(sub_batch,
                                                   [-1, *sub_batch.shape[-3:]])
                            self.metrics_manager.show_image(
                                sub_batch,
                                key,
                                epoch=self.global_epoch,
                                name=name + '_' + few_shot_name[i])
                else:
                    LOGGER.debug('not tuple')
                    if len(
                            batch.shape
                    ) == 4 and batch.shape[-1] <= 4:  # [b, w, h, c], c<4
                        self.metrics_manager.show_image(
                            batch, key, epoch=self.global_epoch, name=name)
