import sys
import tensorflow as tf
from absl import app
from absl import flags
from tf_rlib.utils.ipython import isnotebook
from tf_rlib import blocks, layers, utils, models
    
def run(main):
    try:
        app.run(main)
    except:
        print('Done')

FLAGS = flags.FLAGS

flags.DEFINE_integer('bs', 128, 'Batch Size')
flags.DEFINE_integer(
    'dim', 2,
    'Input Dimensions will decide all the dimensions of operations automatically.'
)
flags.register_validator('dim',
                         lambda value: value in [1, 2, 3],
                         message='--dim must be 1 or 2 or 3')

flags.DEFINE_float('bn_momentum', 0.9, 'momentum for BatchNormalization')
flags.DEFINE_float('bn_epsilon', 1e-5, 'epsilon for BatchNormalization')

flags.DEFINE_string('norm', 'BatchNormalization',
                    'normalization function name')
flags.DEFINE_string('act', 'ReLU', 'activation function name')

if isnotebook():
    sys.argv = sys.argv[:1]  # remove
    run(lambda _:0)