import tensorflow as tf
from tf_rlib import layers, blocks

class UpBlock(blocks.Block):
    '''
    UpSampling + Concat + ResBlock
    '''
    def __init__(self, n_filters, up_size=2, with_concat=True):
        super(UpBlock, self).__init__(n_filters)
        self.with_concat = with_concat
        self.upsampling = layers.UpSampling(up_size=up_size)
        self.concat = tf.keras.layers.Concatenate()
        self.project = blocks.BasicBlock(n_filters,
                                         3,
                                         use_norm=False,
                                         use_act=False)
        self.res_block = blocks.ResBlock(n_filters)

    def call(self, x):
        if self.with_concat:
            x, map_ = x
        x = self.upsampling(x)
        if self.with_concat:
            x = self.concat([x, map_])
            x = self.project(x)
        x = self.res_block(x)
        return x