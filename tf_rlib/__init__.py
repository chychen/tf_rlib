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

# logging
tf.get_logger().setLevel('WARNING')
logging.set_verbosity(logging.INFO)
logging.set_stderrthreshold(logging.INFO)
# envs
current_time = datetime.datetime.now(
    timezone('Asia/Taipei')).strftime("%Y%m%d-%H%M%S")

#############################
########### FLAGS ###########
#############################

FLAGS = flags.FLAGS

# General settings
flags.DEFINE_bool('profile', False, 'use TensorBoard profiler?')
flags.DEFINE_integer('port', '6006', 'port for Tensorbaord')
flags.DEFINE_string('log_path', '/results/{}/log'.format(current_time),
                    'path for logging files')  # save in local is faster
flags.DEFINE_string('save_path', '/results/{}/ckpt'.format(current_time),
                    'path for ckpt files')  # save in local is faster
flags.DEFINE_string('exp_name', 'default', 'name for this experiment')

# Speedup Options
flags.DEFINE_string('gpus', '0', 'os.environ[\'CUDA_VISIBLE_DEVICES\']=?')
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
flags.DEFINE_integer('bs', 128, 'Batch Size')
flags.DEFINE_float('adam_beta_1', 0.9, 'adam beta_1')
flags.DEFINE_float('adam_beta_2', 0.999, 'adam beta_2')
flags.DEFINE_float('adam_epsilon', 1e-8,
                   'adam epsilon, the larger epsilon, the closer to SGD')
## Regularizer
flags.DEFINE_float('l1', 0.0, 'l1 regularizer')
flags.DEFINE_float('l2', 1e-4, 'l2 regularizer')
## Conv
flags.DEFINE_string('kernel_initializer', 'he_normal', 'kernel_initializer')
flags.DEFINE_string('bias_initializer', 'zeros', 'bias_initializer')
flags.DEFINE_string('padding', 'same', 'padding flag for conv, downsample')
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
flags.DEFINE_integer('depth', 3 * 20 + 2, 'depth>=50 use Bottleneck')
flags.DEFINE_bool('bottleneck', True,
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
    logging.info('init flags')

# envs
os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpus
logging.info('CUDA_VISIBLE_DEVICES={}'.format(FLAGS.gpus))

# logging
if not os.path.exists(FLAGS.log_path):
    os.makedirs(FLAGS.log_path)
logging.get_absl_handler().use_absl_log_file(FLAGS.exp_name,
                                             log_dir=FLAGS.log_path)


# new thread for tensorboard, avoiding from annoying logging msg on notebook
def launchTensorBoard():
    os.system('tensorboard --logdir {} --bind_all --port {}'.format(
        FLAGS.log_path, FLAGS.port))
    return


# NOTE: this is a fire-and-forget thread
logging.info(
    'launching Tensorboard at: {} port: {} ... (this is a fire-and-forget thread so no error message if failed)'
    .format(FLAGS.log_path, FLAGS.port))
t = threading.Thread(target=launchTensorBoard, args=([]))
t.start()
