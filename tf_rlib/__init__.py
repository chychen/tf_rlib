import sys
import tensorflow as tf
from absl import app
from absl import flags
from tf_rlib.utils.ipython import isnotebook
from tf_rlib import blocks, layers, utils, models

FLAGS = flags.FLAGS


def define_flags():
    if FLAGS.task == 'Classification':
        pass
    else:
        raise NotImplementedError


def run(main):
    try:
        app.run(main)
    except:
        print('Done')


flags.DEFINE_string('task', 'Classification', 'what is your task?')
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
flags.DEFINE_integer('depth', 3 * 90 + 2, 'depth>=50 use Bottleneck')
flags.DEFINE_bool('bottleneck', True, 'True for ResBottleneck, False for ResBlock')
flags.DEFINE_string('filters_mode', 'small',
                    'small for cifar10, large for imagenet')

if isnotebook():
    sys.argv = sys.argv[:1]  # remove jupyter related arguments
    run(lambda _: 0)

#     flags.register_validator('flag',
#                          lambda v: True,
#                          message='Flag validation failed')
