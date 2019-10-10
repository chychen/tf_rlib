import os
import sys
import threading
import datetime
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
import tensorflow as tf
tf.get_logger().setLevel('WARNING')
from tensorboard import program
from absl import app
from absl import flags, logging
from tf_rlib.utils.ipython import isnotebook
from tf_rlib import blocks, layers, utils, models

# logging
logging.set_verbosity(logging.INFO)
logging.set_stderrthreshold(logging.INFO)

FLAGS = flags.FLAGS

# def define_flags():
#     if FLAGS.task == 'Classification':
#         pass
#     else:
#         raise NotImplementedError


def run(main):
    try:
        app.run(lambda _: 0)
    except:
        print('Done')
    if not os.path.exists(FLAGS.log_path):
        os.makedirs(FLAGS.log_path)
    logging.get_absl_handler().use_absl_log_file(FLAGS.exp_name,
                                                 log_dir=FLAGS.log_path)

    # new thread for tensorboard, avoiding from annoying logging msg on notebook
    def launchTensorBoard():
        os.system('tensorboard --logdir {} --bind_all'.format(FLAGS.log_path))
        return

    t = threading.Thread(target=launchTensorBoard, args=([]))
    t.start()

    main(None)


flags.DEFINE_string('task', 'Classification', 'what is your task?')
flags.DEFINE_string('log_path', '/tmp/{}/log'.format(current_time),
                    'path for logging files')  # save on local is faster
flags.DEFINE_string('save_path', '/tmp/{}/ckpt'.format(current_time),
                    'path for ckpt files')  # save on local is faster
flags.DEFINE_string('exp_name', 'default', 'name for this experiment')
flags.DEFINE_float('lr', 1e-3, 'Initial Learning Rate')
flags.DEFINE_integer('bs', 128, 'Batch Size')
flags.DEFINE_integer('out_dim', 10, 'Model output dimensions')
flags.DEFINE_string('padding', 'same', 'padding flag for conv, downsample')
flags.DEFINE_integer(
    'dim', 2,
    'Input Dimensions will decide all the dimensions of operations automatically.'
)
flags.DEFINE_float('bn_momentum', 0.9, 'momentum for BatchNormalization')
flags.DEFINE_float('bn_epsilon', 1e-5, 'epsilon for BatchNormalization')

flags.DEFINE_string('conv_norm', 'BatchNormalization',
                    'normalization function name')
flags.DEFINE_string('conv_pooling', 'AveragePooling',
                    'pooling function name for shortcut')
flags.DEFINE_string('global_pooling', 'GlobalAveragePooling',
                    'global_pooling function name before dense layer')
flags.DEFINE_string('conv_act', 'ReLU', 'activation function name')

flags.DEFINE_integer(
    'model_alpha', 200,
    '110 layers ranged from 48 to 270 in paper, seems larger is better but parameters inefficiency'
)
flags.DEFINE_integer('depth', 3 * 20 + 2, 'depth>=50 use Bottleneck')
flags.DEFINE_bool('bottleneck', True,
                  'True for ResBottleneck, False for ResBlock')
flags.DEFINE_string('filters_mode', 'small',
                    'small for cifar10, large for imagenet')
# runner
# flags.DEFINE_string('best_state', 'acc',
#                     'runner use best_state to save best ckpt.')
flags.DEFINE_float('adam_beta_1', 0.9, 'adam beta_1')
flags.DEFINE_float('adam_beta_2', 0.999, 'adam beta_2')
flags.DEFINE_float('adam_epsilon', 1e-8,
                   'adam epsilon, the larger epsilon, the closer to SGD')

if isnotebook():
    sys.argv = sys.argv[:1]  # remove jupyter related arguments
    run(lambda _: 0)

#     flags.register_validator('flag',
#                          lambda v: True,
#                          message='Flag validation failed')
