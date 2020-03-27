import tensorflow as tf


class SVDBlur():
    def __init__(self, singular_shape):
        """ For example: cifar image shape=[32, 32, 3]
        then its singular_shape=[3, 32]
        """
        self.ts = tf.Variable(tf.zeros(singular_shape))

    @tf.function
    def blur(self, x, remove):
        x = tf.transpose(x, [2, 0, 1])
        s, u, v = tf.linalg.svd(x)
        self.ts.assign(s)
        self.ts[:, -remove:].assign(tf.zeros_like(self.ts[:, -remove:]))
        blur_x = tf.matmul(
            u, tf.matmul(tf.linalg.diag(self.ts), v, adjoint_b=True))
        blur_x = tf.transpose(blur_x, [1, 2, 0])
        return blur_x
