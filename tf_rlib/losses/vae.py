import tensorflow as tf
from tensorflow.python.keras.losses import LossFunctionWrapper
from tf_rlib.metrics.dice_coef import dice_coef
import numpy as np


def vae_loss(y_true, y_pred):
    """
    y_pred: json, kers are 'z', 'x_logit', 'mean', 'logvar'
    """
    z = y_pred['z']
    x_logit = y_pred['x_logit']
    mean = y_pred['mean']
    logvar = y_pred['logvar']
    x = y_true

    def log_normal_pdf(sample, mean, logvar, raxis=1):
        log2pi = tf.math.log(2. * np.pi)
        return tf.reduce_sum(
            -.5 * ((sample - mean)**2. * tf.exp(-logvar) + logvar + log2pi),
            axis=raxis)

    cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit,
                                                        labels=x)
    logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
    logpz = log_normal_pdf(z, 0., 0.)
    logqz_x = log_normal_pdf(z, mean, logvar)
    elbo = logpx_z + logpz - logqz_x
    return -elbo


class VAELoss(LossFunctionWrapper):
    """ compute the -ELBO (Evidence Lower Bound)
    """
    def __init__(self, name='vae_loss'):
        super(VAELoss, self).__init__(vae_loss,
                                      name=name,
                                      reduction=tf.keras.losses.Reduction.NONE)
