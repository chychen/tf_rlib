import tensorflow as tf
from tensorflow.python.keras.losses import LossFunctionWrapper
from tf_rlib.metrics.dice_coef import dice_coef
import numpy as np


def mse_loss(y_true, y_pred):
    rank = len(y_true.shape)
    return tf.reduce_sum(tf.reduce_mean((y_true - y_pred)**2,
                                        axis=np.arange(1, rank)),
                         axis=None)


class MSELoss(LossFunctionWrapper):
    """ in tf2.0.0, tf.keras.losses.MeanSquaredError only reduce_mean on axis=-1, 
    this class MSELoss reduce_mean on axis=[1,...,rank-1] instead.
    """
    def __init__(self, name='mse_loss'):
        super(MSELoss, self).__init__(mse_loss,
                                      name=name,
                                      reduction=tf.keras.losses.Reduction.NONE)
