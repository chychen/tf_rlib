import tensorflow as tf
from tensorflow.python.keras.metrics import MeanMetricWrapper


class MeanSquaredError(MeanMetricWrapper):
    """ we customize it because most of the time, our output are normalized, therefore we need to scale it back before calculating the metrics.
    """
    def __init__(self,
                 name='MeanAbsoluteError',
                 dtype=None,
                 denorm_fn=lambda x: x):
        super(MeanSquaredError, self).__init__(self.mse(denorm_fn),
                                               name,
                                               dtype=dtype)

    def mse(self, denorm_fn):
        def func(y_true, y_pred):
            denorm_y_true = denorm_fn(y_true)
            denorm_y_pred = denorm_fn(y_pred)
            ret = tf.reduce_mean(
                tf.math.squared_difference(denorm_y_true, denorm_y_pred))
            return ret

        return func
