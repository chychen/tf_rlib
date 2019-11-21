import tensorflow as tf
from tensorflow.python.keras.losses import LossFunctionWrapper
from tf_rlib.metrics.dice_coef import dice_coef


def dice_loss(y_true, y_pred, smooth=1):
    return 1 - dice_coef(y_true, y_pred, smooth)


class DiceLoss(LossFunctionWrapper):
    def __init__(self,
                 reduction=tf.keras.losses.Reduction.NONE,
                 name='dice_loss'):
        super(DiceLoss, self).__init__(dice_loss,
                                       name=name,
                                       reduction=reduction)
