""" Model for Relation Network
"""
import tensorflow as tf
from tf_rlib import layers, blocks, models
from absl import flags
from absl import logging

FLAGS = flags.FLAGS
LOGGER = logging.get_absl_logger()


class RN_Omniglot(models.Model):
    """ Relation Network on Omniglot Dataset
    """
    def __init__(self):
        super(RN_Omniglot, self).__init__()
        self.image_channels = 1
        self.n_channels = 64
        self.embedding_module = self.get_embedding_module()
        self.concat = tf.keras.layers.Concatenate()
        self.relation_module = self.get_relation_module()
        self.sigmoid = tf.keras.layers.Activation(tf.keras.activations.sigmoid,
                                                  name='sigmoid')

    def get_embedding_module(self):
        return self.sequential([
            blocks.BasicBlock(self.n_channels, 3),
            layers.Pooling(2),
            blocks.BasicBlock(self.n_channels, 3),
            layers.Pooling(2),
            blocks.BasicBlock(self.n_channels, 3),
            blocks.BasicBlock(self.n_channels, 3)
        ])

    def get_relation_module(self):
        return self.sequential([
            blocks.BasicBlock(self.n_channels, 3),
            layers.Pooling(2),
            blocks.BasicBlock(self.n_channels, 3),
            layers.Pooling(2),
            tf.keras.layers.Flatten(),
            layers.Dense(8),
            tf.keras.layers.ReLU(),
            layers.Dense(1)
        ])

    def call(self, x):
        """ support_set, query_set = x
        support_set: shape=[bs(batch_size), nq(num_queries/class), c_way, k_shot, 28, 28, 1]
            nq are same
        query_set: shape=[bs(batch_size), nq(num_queries/class), c_way, k_shot, 28, 28, 1] 
            k_shot are same
        """
        support_set, query_set = x
        # reshape
        set_shape = tf.shape(support_set)
        data_shape = set_shape[:4]
        num_data = set_shape[0] * set_shape[1] * set_shape[2] * set_shape[3]
        num_queries = set_shape[0] * set_shape[1] * set_shape[2]
        data_sz = set_shape[4:7]
        support_set = tf.reshape(
            support_set,
            [num_data, data_sz[0], data_sz[1], self.image_channels
             ])  # channel must be set explicitly before conv
        query_set = tf.reduce_mean(query_set, axis=3)
        query_set = tf.reshape(
            query_set,
            [num_queries, data_sz[0], data_sz[1], self.image_channels
             ])  # channel must be set explicitly before conv
        # embedding module
        support_set = self.embedding_module(support_set)
        emb_shape = tf.shape(support_set)
        ## k shot merge by reduce_sum
        support_set = tf.reshape(support_set, [
            data_shape[0], data_shape[1], data_shape[2], data_shape[3],
            emb_shape[-3], emb_shape[-2], emb_shape[-1]
        ])
        support_set = tf.reduce_mean(support_set, axis=3)
        support_set = tf.reshape(
            support_set,
            [num_queries, emb_shape[-3], emb_shape[-2], self.n_channels
             ])  # channel must be set explicitly before conv
        query_set = self.embedding_module(query_set)
        # relation module
        concat = self.concat([support_set, query_set])
        concat = tf.ensure_shape(
            concat,
            (None, 7, 7,
             self.n_channels * 2))  # shape must be set explicitly before dense
        score = self.relation_module(concat)
        # reshape back
        score = tf.reshape(score, data_shape[:3])  # dim for k_shot is reduced
        score = self.sigmoid(score)
        return score
