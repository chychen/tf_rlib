import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow_hub as hub
from tf_rlib.models.research import AEClassifier, Encoder
from tf_rlib.runners.base import runner
from tf_rlib import losses
from absl import flags
from absl import logging
import numpy as np

FLAGS = flags.FLAGS


class OCACNNRunner(runner.Runner):
    """
    Reference: [Active Authentication using an Autoencoder regularized CNN-based One-Class Classifier](https://arxiv.org/abs/1903.01031)
    """
    def __init__(self, train_dataset, valid_dataset=None):
        self.train_dataset = train_dataset
        FLAGS.dim = 2
        super(OCACNNRunner, self).__init__(train_dataset,
                                           valid_dataset=valid_dataset,
                                           best_state='auc_roc')

    def init(self):
        input_shape = self.train_dataset.element_spec[0].shape[1:]
        #         # Pretrained
        self.encoder = tf.keras.Sequential([
            hub.KerasLayer(
                'https://tfhub.dev/google/imagenet/resnet_v2_50/feature_vector/4',
                input_shape=input_shape)
        ])
        self.encoder.trainable = False  # frozen
        #         self.encoder = Encoder()
        self.model = AEClassifier()
        train_metrics = {
            'loss': tf.keras.metrics.Mean('loss'),
            'mae_loss': tf.keras.metrics.Mean('mae_loss'),
            'bce_loss': tf.keras.metrics.Mean('bce_loss'),
            'acc': tf.keras.metrics.BinaryAccuracy('acc')
        }
        valid_metrics = {
            'acc':
            tf.keras.metrics.BinaryAccuracy('acc'),
            'mae_loss':
            tf.keras.metrics.Mean('mae_loss'),
            'auc_roc':
            tf.keras.metrics.AUC(num_thresholds=200,
                                 curve='ROC',
                                 name='auc_roc'),
            'auc_pr':
            tf.keras.metrics.AUC(num_thresholds=200, curve='PR', name='auc_pr')
        }
        self.bce = tf.keras.losses.BinaryCrossentropy(
            from_logits=True,
            reduction=tf.keras.losses.Reduction.NONE)  # distributed-aware
        self.mae = losses.MAELoss()
        self.optim = tfa.optimizers.AdamW(weight_decay=FLAGS.wd,
                                          lr=0.0,
                                          beta_1=FLAGS.adam_beta_1,
                                          beta_2=FLAGS.adam_beta_2,
                                          epsilon=FLAGS.adam_epsilon)
        return {
            'Encoder': self.encoder,
            'AEClassifier': self.model
        }, {
            'Encoder': input_shape,
            'AEClassifier': self.encoder.output_shape[1:]
        }, train_metrics, valid_metrics


#         return {'Encoder': self.encoder, 'AEClassifier': self.model}, {'Encoder': (224, 224, 3), 'AEClassifier': (14,14,512,)}, train_metrics, valid_metrics

    def begin_fit_callback(self, lr):
        self.init_lr = lr
        self.lr_scheduler = tf.keras.experimental.CosineDecay(
            self.init_lr, None)

    def begin_epoch_callback(self, epoch_id, epochs):
        self.lr_scheduler.decay_steps = epochs
        self.optim.learning_rate = self.lr_scheduler(epoch_id)

        self.log_scalar('lr', self.optim.learning_rate, training=True)
        self.optim.weight_decay = FLAGS.wd * self.optim.learning_rate / self.init_lr
        self.log_scalar('wd', self.optim.weight_decay, training=True)

    def train_step(self, x, y):
        """
        Args:
            x (batch): mini batch data
            y (batch): mini batch label
        Returns:
            losses (dict)
        """
        with tf.GradientTape() as tape:
            noise_mean = 0.0
            noise_stddev = 1e-2
            lambda_loss = 1.0
            embedding = self.encoder(x,
                                     training=self.encoder.trainable)  # frozen
            target_probs, target_recs = self.model(embedding, training=True)
            others_probs, others_recs = self.model(tf.random.normal(
                (FLAGS.bs, 2048), noise_mean, noise_stddev),
                                                   training=True)
            #             embedding = self.encoder(x, training=True)
            #             target_probs, target_recs = self.model(embedding, training=True)
            #             others_probs, others_recs = self.model(tf.random.normal((FLAGS.bs, 14,14,512), noise_mean, noise_stddev), training=True)
            logits = tf.concat([target_probs, others_probs], axis=0)
            probs = tf.math.sigmoid(logits)
            labels = tf.concat([y, tf.zeros((FLAGS.bs, 1), tf.float32)],
                               axis=0)

            bce_loss = self.bce(labels, probs)
            mae_loss = self.mae(x, target_recs)
            # because noise couldn't be computed by reconstruction loss(mae)
            mae_loss = tf.tile(mae_loss, [2])
            loss = bce_loss + lambda_loss * mae_loss

            loss = tf.nn.compute_average_loss(loss,
                                              global_batch_size=FLAGS.bs *
                                              2)  # distributed-aware
            regularization_loss = tf.nn.scale_regularization_loss(
                tf.math.add_n(self.encoder.losses +
                              self.model.losses))  # distributed-aware

            total_loss = loss + regularization_loss

        grads = tape.gradient(
            total_loss,
            self.model.trainable_weights + self.encoder.trainable_weights)

        self.optim.apply_gradients(
            zip(grads,
                self.model.trainable_weights + self.encoder.trainable_weights))
        return {
            'loss': [loss],
            'acc': [labels, probs],
            'mae_loss': [tf.reduce_mean(mae_loss)],
            'bce_loss': [tf.reduce_mean(bce_loss)]
        }

    def validate_step(self, x, y):
        """
        Args:
            x (batch): mini batch data
            y (batch): mini batch label
        Returns:
            metrics (dict)
        """
        embedding = self.encoder(x, training=False)
        logits, reconstruct = self.model(embedding, training=False)
        probs = tf.math.sigmoid(logits)
        mae_loss = self.mae(x, reconstruct)
        return {
            'acc': [y, probs],
            'mae_loss': [tf.reduce_mean(mae_loss)],
            'auc_roc': [y, probs],
            'auc_pr': [y, probs]
        }

    def custom_log_data(self, x_batch, y_batch):
        embedding = self.encoder(x_batch, training=False)
        _, reconstruct = self.model(embedding, training=False)
        return 'reconstructs', reconstruct

    @property
    def required_flags(self):
        return ['bs']

    @property
    def support_amp(self):
        return False
