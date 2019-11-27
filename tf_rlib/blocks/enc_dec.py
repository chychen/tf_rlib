import tensorflow as tf
from tf_rlib import layers, blocks


class EncDecBlock(blocks.Block):
    def __init__(self, filters, mid_block=None):
        super(EncDecBlock, self).__init__(filters)
        self.enc_block = blocks.ResBlock(filters, strides=2)
        self.mid_block = mid_block if mid_block is not None else lambda in_: in_
        self.dec_block = blocks.UpBlock(filters // 2,
                                        with_concat=False,
                                        up_size=2)

    def call(self, x):
        x = self.enc_block(x)
        x = self.mid_block(x)
        x = self.dec_block(x)
        return x