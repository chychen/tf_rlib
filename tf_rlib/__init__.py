import os
import sys
import threading
from pytz import timezone
import datetime
import tensorflow as tf
from absl import app
from absl import flags, logging
from tf_rlib.utils.ipython import isnotebook
from tf_rlib import blocks, datasets, layers, models, runners, utils, metrics, losses
from tf_rlib.utils import init_tf_rlib

# envs
current_time = datetime.datetime.now(
    timezone('Asia/Taipei')).strftime("%Y%m%d-%H%M%S")

#############################
########### FLAGS ###########
#############################

FLAGS = flags.FLAGS
LOGGER = logging.get_absl_logger()

# General settings
flags.DEFINE_string('log_level', 'INFO',
                    'log_level: DEBUG, INFO, WARNING, ERROR')
flags.DEFINE_bool('tqdm', True, 'use tqdm?')
flags.DEFINE_bool('profile', False, 'use TensorBoard profiler?')
flags.DEFINE_bool('purge_logs', False, 'remove all logs')
flags.DEFINE_string('current_time', current_time,
                    'timezone: Asia/Taipei, strftime: %Y%m%d-%H%M%S')
flags.DEFINE_string('local_path', '/results', 'tmp folder')
flags.DEFINE_string('log_path', None, 'path for logging files')
flags.DEFINE_string('path_postfix', '', 'postfix after log_path')
flags.DEFINE_string('save_path', None, 'path for ckpt files')
flags.DEFINE_string('exp_name', 'default', 'name for this experiment')
flags.DEFINE_string('comment', None, 'any comment?')
flags.DEFINE_string('benchmark', None, 'class name of benchmark runner')

# Speedup Options
flags.DEFINE_string(
    'gpus', None,
    'default None means all, os.environ[\'CUDA_VISIBLE_DEVICES\']=?')
flags.DEFINE_integer('num_gpus', None, 'num_gpus')
flags.DEFINE_bool('amp', False, 'use Automatically Mixed Precision?')
flags.DEFINE_bool(
    'xla', False,
    'use XLA compiler, 99% models will get speedup, big model might need large compiling time.'
)

# I/O
flags.DEFINE_integer('out_dim', None, 'Model output dimensions')
flags.DEFINE_integer(
    'dim', None,
    'Input Dimensions will decide all the dimensions of operations automatically.'
)
flags.DEFINE_integer('pre_augment', None,
                     'how many times to pre-augment the dataset')

# General Hyper-perameters

## Optimizer
flags.DEFINE_integer('bs', None, 'global Batch Size for all gpus')
flags.DEFINE_float('lr', None, 'Initial Learning Rate')
flags.DEFINE_integer('epochs', None, 'number of epochs')
flags.DEFINE_integer(
    'steps_per_epoch', None,
    'number of steps per epoch, used when one epoch is too large')
flags.DEFINE_integer('warmup', 5, 'number of epochs for warming up')
flags.DEFINE_float('adam_beta_1', 0.9, 'adam beta_1')
flags.DEFINE_float('adam_beta_2', 0.999, 'adam beta_2')
flags.DEFINE_float('adam_epsilon', 1e-8,
                   'adam epsilon, the larger epsilon, the closer to SGD')
## LR Scheduler TODO early stop
flags.DEFINE_integer('lr_factor', None, '')
flags.DEFINE_float('lr_patience', None, '')

## Regularizer
flags.DEFINE_float('l1', 0.0, 'l1 regularizer')
flags.DEFINE_float('l2', 0.0, 'l2 regularizer')
flags.DEFINE_float('wd', 0.0,
                   'weight decay in correct way, such as AdamW, SGDW')
## Conv
flags.DEFINE_string('kernel_initializer', 'he_normal',
                    'kernel_initializer, such as [he_normal, glorot_uniform]')
flags.DEFINE_string('bias_initializer', 'zeros', 'bias_initializer')
flags.DEFINE_string(
    'padding', 'same',
    'same or valid or same_symmetric, padding flag for transpose/conv, up/downsample'
)
flags.DEFINE_string('conv_act', 'ReLU', 'activation function name')
flags.DEFINE_string('interpolation', 'nearest', 'method for UpSampling layer')
## BN
flags.DEFINE_string('conv_norm', 'BatchNormalization',
                    'normalization function name')
flags.DEFINE_float('bn_momentum', 0.9, 'momentum for BatchNormalization')
flags.DEFINE_float('bn_epsilon', 1e-5, 'epsilon for BatchNormalization')
flags.DEFINE_integer(
    'groups', None,
    'if your norm is GroupNormalization, please set this parameter, and check your groups can be divisible by all channel numbers using GroupNormalization'
)
## Pooling
flags.DEFINE_string('conv_pooling', 'MaxPooling',
                    'pooling for downsampling, encoding features.')
flags.DEFINE_string('shortcut_pooling', 'AveragePooling',
                    'pooling function name for shortcut')
flags.DEFINE_string('global_pooling', 'GlobalAveragePooling',
                    'global_pooling function name before dense layer')

# Model Architecture
flags.DEFINE_integer(
    'model_alpha', 200,
    '110 layers ranged from 48 to 270 in paper, seems larger is better but parameters inefficiency, if FLAGS.amp=True, layers_per_block = 3 if FLAGS.bottleneck else 2, total_blocks=(FLAGS.depth-2)/layers_per_block, please set FLAGS.model_alpha=total_blocks*8 to make sure channels are equal to multiple of 8.'
)
flags.DEFINE_integer('depth', None, 'depth>=50 use Bottleneck')
flags.DEFINE_bool('bottleneck', None,
                  'True for ResBottleneck, False for ResBlock')
flags.DEFINE_string('filters_mode', 'small',
                    'small for cifar10, large for imagenet')

#############################
########## RUNNERS ##########
#############################

# General Settings (default is None, set value within each runner)
flags.DEFINE_string(
    'runner', None,
    'FLAGS.runner are set as the name of the running runner in the template <runner.py> .'
)
flags.DEFINE_string(
    'loss_fn', None,
    'go check recommended loss functions in runner API, such as tf.rlib.runners.ADAERunner.LOSSES_POOL'
)

# ClassificationRunner

# SegmentationRunner

# ADAERunner (Anomaly Detection AutoEncoder Runner)
flags.DEFINE_integer('latent_dim', None, 'dims of latent space')

# FewShotRunner
flags.DEFINE_integer('c_way', None, 'c_way')
flags.DEFINE_integer('k_shot', None, 'k_shot')

# OodSvdRndRunner
flags.DEFINE_integer('svd_remove', None,
                     'how many singular value reduced to zeros?')

#############################
########### RLIBS ###########
#############################

# remove jupyter related arguments
if isnotebook():
    sys.argv = sys.argv[:1]

# init FLAGS
try:
    app.run(lambda _: 0)
except:
    LOGGER.info('init flags')

# init env settings
init_tf_rlib(first=True)
