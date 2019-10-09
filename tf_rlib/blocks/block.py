import tensorflow as tf
from tf_rlib import layers


class Block(tf.keras.layers.Layer):
    def __init__(self, filters, strides=1):
        super(Block, self).__init__()
