import tensorflow as tf
from tensorflow.keras import layers
from absl import flags

FLAGS = flags.FLAGS


class Act(tf.keras.layers.Layer):
    def __init__(self, act_type=None):
        """ By default with act_type=None, the API is controled by FLAGS.conv_act.
        """
        super(Act, self).__init__()
        if act_type is None:
            act_type = FLAGS.conv_act

        found = False
        for classname in layers.__dict__:
            if classname.lower() == act_type.lower():
                found = True
                act_op = layers.__dict__[classname]
                break
        if not found:
            for classname in tf.math.__dict__:
                if classname.lower() == act_type.lower():
                    found = True
                    act_op = tf.math.__dict__[classname]
                    break

        if act_type.lower() == 'leakyrelu':
            self.act_op = act_op(alpha=0.2)
        elif act_type.lower() == 'relu':
            self.act_op = act_op()
        elif act_type.lower() == 'sigmoid':
            self.act_op = layers.Activation(act_op)
        elif act_type.lower() == 'tanh':
            self.act_op = layers.Activation(act_op)
        else:
            raise ValueError

    def call(self, x):
        x = self.act_op(x)
        return x
