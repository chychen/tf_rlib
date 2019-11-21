import tensorflow as tf
from tf_rlib import layers, blocks

class DownBlock(blocks.Block):
    '''
    Pooling + resBlock
    '''
    def __init__(self, n_filters, pool_size=2):
        super(DownBlock, self).__init__(n_filters)
        self.pool = layers.Pooling(pool_size=pool_size)
        self.res_block = blocks.ResBlock(n_filters)

    def call(self, x):
        x = self.pool(x)
        x = self.res_block(x)
        return x