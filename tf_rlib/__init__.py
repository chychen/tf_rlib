import os
import sys
import threading
from pytz import timezone
import datetime
import tensorflow as tf
from absl import app
from absl import flags, logging
from tf_rlib.utils.ipython import isnotebook
from tf_rlib import blocks, datasets, layers, models, research, runners, utils
from tf_rlib.utils import purge_logs

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
flags.DEFINE_bool('profile', False, 'use TensorBoard profiler?')
flags.DEFINE_bool('purge_logs', False, 'remove all logs')
flags.DEFINE_string('current_time', current_time,
                    'timezone: Asia/Taipei, strftime: %Y%m%d-%H%M%S')
# flags.DEFINE_integer('port', '6006', 'port for Tensorbaord')
flags.DEFINE_string(
    'local_path', '/results',
    'tmp folder')  # NOTE: save in local is faster than mounted location
flags.DEFINE_string('log_path', 'log', 'path for logging files')
flags.DEFINE_string('save_path', 'ckpt', 'path for ckpt files')
flags.DEFINE_string('exp_name', 'default', 'name for this experiment')
flags.DEFINE_string('comment', None, 'any comment?')
flags.DEFINE_string('benchmark_runner', None, 'any comment?')

# Speedup Options
flags.DEFINE_string('gpus', '0,1,2,3,4,5,6,7',
                    'os.environ[\'CUDA_VISIBLE_DEVICES\']=?')
flags.DEFINE_bool('amp', False, 'use Automatically Mixed Precision?')

# I/O
flags.DEFINE_integer('out_dim', 10, 'Model output dimensions')
flags.DEFINE_integer(
    'dim', 2,
    'Input Dimensions will decide all the dimensions of operations automatically.'
)

# General Hyper-perameters
## Optimizer
flags.DEFINE_float('lr', 1e-3, 'Initial Learning Rate')
flags.DEFINE_integer('epochs', 300, 'number of epochs for warming up')
flags.DEFINE_integer('warmup', 5, 'number of epochs for warming up')
flags.DEFINE_integer('bs', 128, 'global Batch Size for all gpus')
flags.DEFINE_float('adam_beta_1', 0.9, 'adam beta_1')
flags.DEFINE_float('adam_beta_2', 0.999, 'adam beta_2')
flags.DEFINE_float('adam_epsilon', 1e-8,
                   'adam epsilon, the larger epsilon, the closer to SGD')
## Regularizer
flags.DEFINE_float('l1', 0.0, 'l1 regularizer')
flags.DEFINE_float('l2', 1e-4, 'l2 regularizer')
## Conv
flags.DEFINE_string('kernel_initializer', 'he_normal',
                    'kernel_initializer, such as [he_normal, glorot_uniform]')
flags.DEFINE_string('bias_initializer', 'zeros', 'bias_initializer')
flags.DEFINE_string('padding', 'same',
                    'same or valid, padding flag for conv, downsample')
flags.DEFINE_string('conv_act', 'ReLU', 'activation function name')
## BN
flags.DEFINE_string('conv_norm', 'BatchNormalization',
                    'normalization function name')
flags.DEFINE_float('bn_momentum', 0.9, 'momentum for BatchNormalization')
flags.DEFINE_float('bn_epsilon', 1e-5, 'epsilon for BatchNormalization')
## Pooling
flags.DEFINE_string('conv_pooling', 'AveragePooling',
                    'pooling function name for shortcut')
flags.DEFINE_string('global_pooling', 'GlobalAveragePooling',
                    'global_pooling function name before dense layer')

# Model Architecture
flags.DEFINE_integer(
    'model_alpha', 200,
    '110 layers ranged from 48 to 270 in paper, seems larger is better but parameters inefficiency'
)
flags.DEFINE_integer('depth', None, 'depth>=50 use Bottleneck')
flags.DEFINE_bool('bottleneck', None,
                  'True for ResBottleneck, False for ResBlock')
flags.DEFINE_string('filters_mode', 'small',
                    'small for cifar10, large for imagenet')

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

# rename log/save path
FLAGS.log_path = os.path.join(FLAGS.local_path, FLAGS.exp_name,
                              FLAGS.current_time, FLAGS.log_path)
FLAGS.save_path = os.path.join(FLAGS.local_path, FLAGS.exp_name,
                               FLAGS.current_time, FLAGS.save_path)
# logging config
if FLAGS.purge_logs:
    purge_logs()

if not os.path.exists(FLAGS.log_path):
    os.makedirs(FLAGS.log_path)

hd = logging.logging.FileHandler(os.path.join(FLAGS.log_path, 'logging.txt'))
hd.setLevel('INFO')
LOGGER.addHandler(hd)
tf.get_logger().setLevel('WARNING')
LOGGER.setLevel(FLAGS.log_level)

## show all FLAGS
for flag, value in FLAGS.flag_values_dict().items():
    LOGGER.info('FLAGS: --{}={}'.format(flag, value))

# envs
os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpus

# # new thread for tensorboard, avoiding from annoying logging msg on notebook
# def launchTensorBoard():
#     os.system('tensorboard --logdir {} --bind_all --port {}'.format(
#         FLAGS.log_path, FLAGS.port))
#     return

# # NOTE: this is a fire-and-forget thread
# logging.info('launching tensorboard --logdir {} --bind_all --port {}'.format(
#     FLAGS.log_path, FLAGS.port))
# t = threading.Thread(target=launchTensorBoard, args=([]))
# t.start()
