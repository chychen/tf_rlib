import tensorflow as tf
from tensorflow.python.keras.losses import LossFunctionWrapper


class CustomLoss(LossFunctionWrapper):
    def __init__(self,
                 loss_fn,
                 reduction=tf.keras.losses.Reduction.NONE,
                 name='custom_loss'):
        super(CustomLoss, self).__init__(loss_fn,
                                         name=name,
                                         reduction=reduction)
