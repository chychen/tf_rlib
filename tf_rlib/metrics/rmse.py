import numpy as np
import tensorflow as tf
from tensorflow.python.keras.metrics import MeanMetricWrapper


class RootMeanSquaredError(MeanMetricWrapper):
    """ we customize it because most of the time, our output are normalized, therefore we need to scale it back before calculating the metrics.
    """
    def __init__(self,
                 name='RootMeanSquaredError',
                 dtype=None,
                 denorm_fn=lambda x: x):
        super(RootMeanSquaredError, self).__init__(self.rmse(denorm_fn),
                                                   name,
                                                   dtype=dtype)

    def rmse(self, denorm_fn):
        def func(y_true, y_pred):
            rank = len(y_true.shape)
            denorm_y_true = denorm_fn(y_true)
            denorm_y_pred = denorm_fn(y_pred)
            ret = tf.reduce_mean(tf.math.squared_difference(
                denorm_y_true, denorm_y_pred),
                                 axis=np.arange(1, rank))
            return tf.math.sqrt(ret)

        return func
