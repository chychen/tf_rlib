import numpy as np
import tensorflow as tf
from tensorflow.python.keras.metrics import MeanMetricWrapper


class RULHitRate(MeanMetricWrapper):
    """ 
    """
    def __init__(self,
                 name='RULHitRate',
                 dtype=None,
                 target_dim=0,
                 margin=None,
                 denorm_fn=lambda x: x):
        super(RULHitRate, self).__init__(self.hitrate(target_dim, margin,
                                                      denorm_fn),
                                         name,
                                         dtype=dtype)

    def hitrate(self, target_dim, margin, denorm_fn):
        def func(y_true, y_pred):
            denorm_y_true = denorm_fn(y_true)
            denorm_y_pred = denorm_fn(y_pred)
            if target_dim is not None:
                mae = tf.abs(denorm_y_true[:, target_dim] -
                             denorm_y_pred[:, target_dim])
            else:
                mae = tf.abs(denorm_y_true - denorm_y_pred)
            hit_rate = tf.reduce_mean(tf.cast(mae < margin, dtype=tf.float32))
            return hit_rate

        return func
