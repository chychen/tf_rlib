import numpy as np
import tensorflow as tf
from tensorflow.python.keras.metrics import MeanMetricWrapper


class MeanAbsoluteError(MeanMetricWrapper):
    """ we customize it because most of the time, our output are normalized, therefore we need to scale it back before calculating the metrics.
    """
    def __init__(self,
                 name='MeanAbsoluteError',
                 dtype=None,
                 denorm_fn=lambda x: x):
        super(MeanAbsoluteError, self).__init__(self.mae(denorm_fn),
                                                name,
                                                dtype=dtype)

    def mae(self, denorm_fn):
        def func(y_true, y_pred):
            rank = len(y_true.shape)
            denorm_y_true = denorm_fn(y_true)
            denorm_y_pred = denorm_fn(y_pred)
            ret = tf.reduce_mean(tf.abs(denorm_y_true - denorm_y_pred),
                                 axis=np.arange(1, rank))
            return ret

        return func
