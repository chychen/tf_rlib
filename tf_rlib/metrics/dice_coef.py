import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.python.keras.metrics import MeanMetricWrapper


def dice_coef(y_true, y_pred, smooth=1):
    """
    Dice = (2*|X & Y|)/ (|X|+ |Y|)
         =  2*sum(|A*B|)/(sum(A^2)+sum(B^2))
    ref: https://arxiv.org/pdf/1606.04797v1.pdf
    """
    bs = tf.shape(y_true)[0]
    # flatten
    y_true_flat = tf.reshape(y_true, [bs, -1])
    y_pred_flat = tf.reshape(y_pred, [bs, -1])
    intersection = K.sum(K.abs(y_true_flat * y_pred_flat), axis=-1)
    return (2. * intersection + smooth) / (K.sum(
        K.square(y_true_flat), -1) + K.sum(K.square(y_pred_flat), -1) + smooth)


class DiceCoefficient(MeanMetricWrapper):
    def __init__(self, name='dice_coef', dtype=None):
        super(DiceCoefficient, self).__init__(dice_coef, name, dtype=dtype)