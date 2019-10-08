import tensorflow as tf
from tensorflow.keras import layers
from absl import flags

FLAGS = flags.FLAGS


class Act(tf.keras.layers.Layer):
    def __init__(self):
        super(Act, self).__init__()
        act_op = layers.__dict__[FLAGS.conv_act]
        if FLAGS.conv_act == 'LeakyReLU':
            self.act_op = act_op(alpha=0.2)
        elif FLAGS.conv_act == 'ReLU':
            self.act_op = act_op()
        else:
            raise ValueError

    def call(self, x):
        x = self.act_op(x)
        return x
