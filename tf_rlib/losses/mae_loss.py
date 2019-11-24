import tensorflow as tf
from tensorflow.python.keras.losses import LossFunctionWrapper
from tf_rlib.metrics.dice_coef import dice_coef
import numpy as np


def mae_loss(y_true, y_pred):
    rank = len(y_true.shape)
    return tf.reduce_sum(tf.reduce_mean(tf.abs(y_true - y_pred),
                                        axis=np.arange(1, rank)),
                         axis=None)


class MAELoss(LossFunctionWrapper):
    """ in tf2.0.0, tf.keras.losses.MAE only reduce_mean on axis=-1, 
    this class MAELoss reduce_mean on axis=[1,...,rank-1] instead.
    """
    def __init__(self, name='mse_loss'):
        super(MAELoss, self).__init__(mae_loss,
                                      name=name,
                                      reduction=tf.keras.losses.Reduction.NONE)
