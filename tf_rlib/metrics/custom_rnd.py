import tensorflow as tf
from absl import flags, logging
import io
import matplotlib.pyplot as plt

FLAGS = flags.FLAGS


class RNDMetrics(tf.keras.metrics.Metric):
    MODE_FIGURE = 'figure'
    MODE_TNR95TPR = 'tnr@95tpr'

    def __init__(self, mode, amount, num_threshold=100, **kwargs):
        """
        mode(str):
            - RNDMetrics.MODE_FIGURE: TPR-TNR image
            - RNDMetrics.MODE_TNR95TPR: tnr@95tpr value
        """
        super(RNDMetrics, self).__init__(name=mode, **kwargs)
        self.mode = mode
        self.num_threshold = num_threshold
        self.all_scores = self.add_weight(name='all_scores',
                                          shape=(amount, ),
                                          initializer='zeros',
                                          dtype=tf.float32)
        self.all_labels = self.add_weight(name='all_labels',
                                          shape=(amount, ),
                                          initializer='zeros',
                                          dtype=tf.bool)
        self.counter = self.add_weight(name='counter',
                                       shape=(),
                                       initializer='zeros',
                                       dtype=tf.int32)
        self.tp = self.add_weight(name='tp',
                                  shape=(num_threshold, ),
                                  initializer='zeros',
                                  dtype=tf.int64)
        self.tn = self.add_weight(name='tn',
                                  shape=(num_threshold, ),
                                  initializer='zeros',
                                  dtype=tf.int64)
        self.fp = self.add_weight(name='fp',
                                  shape=(num_threshold, ),
                                  initializer='zeros',
                                  dtype=tf.int64)
        self.fn = self.add_weight(name='fn',
                                  shape=(num_threshold, ),
                                  initializer='zeros',
                                  dtype=tf.int64)

        self.tpr = self.add_weight(name='tpr',
                                   shape=(num_threshold, ),
                                   initializer='zeros',
                                   dtype=tf.float32)
        self.tnr = self.add_weight(name='tnr',
                                   shape=(num_threshold, ),
                                   initializer='zeros',
                                   dtype=tf.float32)

    def update_state(self, y_true, y_pred, sample_weight=None):
        size = tf.shape(y_true)[0]
        self.all_scores[self.counter:self.counter + size].assign(y_pred[:, 0])
        self.all_labels[self.counter:self.counter + size].assign(
            tf.cast(y_true[:, 0], tf.bool))
        self.counter.assign_add(size)

    def result(self):
        max_score = tf.math.reduce_max(self.all_scores)
        min_score = tf.math.reduce_min(self.all_scores)
        stepsize = (max_score - min_score) / (self.num_threshold - 1)

        is_ok = tf.math.logical_not(self.all_labels)
        is_ng = self.all_labels
        for i in range(self.num_threshold):
            ths = min_score + stepsize * i
            self.tp[i].assign(
                tf.math.count_nonzero(
                    tf.math.logical_and(self.all_scores >= ths, is_ng)))
            self.tn[i].assign(
                tf.math.count_nonzero(
                    tf.math.logical_and(self.all_scores < ths, is_ok)))
            self.fp[i].assign(
                tf.math.count_nonzero(
                    tf.math.logical_and(self.all_scores >= ths, is_ok)))
            self.fn[i].assign(
                tf.math.count_nonzero(
                    tf.math.logical_and(self.all_scores < ths, is_ng)))

        self.tpr.assign(
            tf.cast(self.tp, tf.float32) /
            tf.cast(self.tp + self.fn, tf.float32))
        self.tnr.assign(
            tf.cast(self.tn, tf.float32) /
            tf.cast(self.tn + self.fp, tf.float32))

        tn_95tp = self.tnr[tf.argmin(tf.math.abs(self.tpr - 0.95))]
        closest95tp = self.tpr[tf.argmin(tf.math.abs(self.tpr - 0.95))]
        if self.mode == RNDMetrics.MODE_FIGURE:
            title = '{:.2f}% TNR @ {:.2f}% TPR\nAverage TPR:{:.2f}\nAverage TNR:{:.2f}'.format(
                tn_95tp * 100., closest95tp * 100.,
                tf.math.reduce_mean(self.tpr) * 100.,
                tf.math.reduce_mean(self.tnr) * 100.)

            figure = plt.figure(figsize=(5, 5))
            plt.plot(self.tnr.numpy(),
                     self.tpr.numpy(),
                     marker='.',
                     label='TPR-TNR')
            plt.suptitle(title, y=1.0)
            plt.xlabel('True Negative Rate')
            plt.ylabel('True Positive Rate')
            plt.legend()

            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            # Closing the figure prevents it from being displayed directly inside
            # the notebook.
            plt.close(figure)
            buf.seek(0)
            # Convert PNG buffer to TF image
            image = tf.image.decode_png(buf.getvalue(), channels=4)
            # Add the batch dimension
            image = tf.expand_dims(image, 0)
            return image
        elif self.mode == RNDMetrics.MODE_TNR95TPR:
            return tn_95tp

    def reset_states(self):
        self.all_scores.assign(tf.zeros_like(self.all_scores))
        self.all_labels.assign(tf.zeros_like(self.all_labels))
        self.counter.assign(tf.zeros_like(self.counter))

        self.tp.assign(tf.zeros_like(self.tp))
        self.tn.assign(tf.zeros_like(self.tn))
        self.fp.assign(tf.zeros_like(self.fp))
        self.fn.assign(tf.zeros_like(self.fn))

        self.tpr.assign(tf.zeros_like(self.tpr))
        self.tnr.assign(tf.zeros_like(self.tnr))
