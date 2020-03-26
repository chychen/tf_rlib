import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.python.keras.losses import LossFunctionWrapper
from tf_rlib.metrics.dice_coef import dice_coef
import numpy as np


def log_normal_pdf(sample, mean, logvar, raxis=1):
    log2pi = tf.math.log(2. * np.pi)
    return tf.reduce_sum(
        -.5 * ((sample - mean)**2. * tf.exp(-logvar) + logvar + log2pi),
        axis=raxis)


class VAEGaussLoss(LossFunctionWrapper):
    """ compute the -ELBO (Evidence Lower Bound)
    """
    def __init__(self, name='vae_loss'):
        super(VAEGaussLoss,
              self).__init__(self.vae_gaussian_loss,
                             name=name,
                             reduction=tf.keras.losses.Reduction.NONE)

    def vae_gaussian_loss(self, y_true, y_pred):
        """
        y_pred: json, kers are 'z', 'x_logit', 'mean', 'logvar'
        """
        z = y_pred['z']
        x_logit = y_pred['x_logit']
        mean = y_pred['mean']
        logvar = y_pred['logvar']
        x = y_true

        reconstruct_term = tfp.distributions.MultivariateNormalDiag(
            tf.keras.layers.Flatten()(x_logit),
            scale_identity_multiplier=0.05).log_prob(
                tf.keras.layers.Flatten()(x))

        logpx_z = reconstruct_term
        logpz = log_normal_pdf(z, 0., 0.)
        logqz_x = log_normal_pdf(z, mean, logvar)
        elbo = logpx_z + logpz - logqz_x
        return -elbo


class VAEBernoulliLoss(LossFunctionWrapper):
    """ compute the -ELBO (Evidence Lower Bound)
    """
    def __init__(self, name='vae_loss'):
        super(VAEBernoulliLoss,
              self).__init__(self.vae_bernoulli_loss,
                             name=name,
                             reduction=tf.keras.losses.Reduction.NONE)

    def vae_bernoulli_loss(self, y_true, y_pred):
        """
        y_pred: json, kers are 'z', 'x_logit', 'mean', 'logvar'
        """
        z = y_pred['z']
        x_logit = y_pred['x_logit']
        mean = y_pred['mean']
        logvar = y_pred['logvar']
        x = y_true

        reconstruct_term = tf.nn.sigmoid_cross_entropy_with_logits(
            logits=x_logit, labels=x)
        logpx_z = -tf.reduce_sum(reconstruct_term, axis=[1, 2, 3])
        logpz = log_normal_pdf(z, 0., 0.)
        logqz_x = log_normal_pdf(z, mean, logvar)
        elbo = logpx_z + logpz - logqz_x
        return -elbo
