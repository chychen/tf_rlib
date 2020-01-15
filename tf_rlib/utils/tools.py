import os
import shutil
import tensorflow as tf
from absl import flags, logging
from tensorflow.keras.mixed_precision import experimental as mixed_precision

FLAGS = flags.FLAGS
LOGGER = logging.get_absl_logger()


def purge_logs():
    shutil.rmtree(FLAGS.local_path, ignore_errors=True)
    logging.warn('purged all logs under :{}'.format(FLAGS.local_path))


def set_logging(level):
    FLAGS.log_level = level
    LOGGER.setLevel(FLAGS.log_level)


def set_gpus(gpus):
    FLAGS.gpus = gpus
    os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpus
    logging.warn('update CUDA_VISIBLE_DEVICES={}'.format(FLAGS.gpus))


def reset_paths():
    # rename log/save path
    if FLAGS.log_path is None:
        FLAGS.log_path = os.path.join(FLAGS.local_path, FLAGS.exp_name,
                                      FLAGS.current_time, 'log')
        if not os.path.exists(FLAGS.log_path):
            os.makedirs(FLAGS.log_path)
        LOGGER.warn('update FLAGS: --{}={}'.format('log_path', FLAGS.log_path))
    if FLAGS.save_path is None:
        FLAGS.save_path = os.path.join(FLAGS.local_path, FLAGS.exp_name,
                                       FLAGS.current_time, 'ckpt')
        LOGGER.warn('update FLAGS: --{}={}'.format('save_path',
                                                   FLAGS.save_path))


def set_exp_name(name):
    FLAGS.exp_name = name
    reset_paths()


def set_xla(enable):
    FLAGS.xla = enable
    tf.keras.backend.clear_session()
    tf.config.optimizer.set_jit(FLAGS.xla)


def set_amp(enable):
    FLAGS.amp = enable
    if FLAGS.amp:
        policy = mixed_precision.Policy('mixed_float16')
        LOGGER.info(
            'Kindly Reminder: mixed_precision enables you train model with double batch size and learning rate!!!'
        )
        LOGGER.info(
            'General rules of thumb: Dimensions (batch, channels, image size, dense nodes) in multiples of 8 If not, Tensor Cores probably still work, but might involve padding (less efficient) Dimensions < 256, use power of 2 Batch size (depending on model) might be optimal, please ref: https://docs.nvidia.com/deeplearning/sdk/dl-performance-guide/index.html#perf-guidelines'
        )
    else:
        policy = mixed_precision.Policy('float32')

    mixed_precision.set_policy(policy)
    LOGGER.warn('Compute dtype: {}'.format(policy.compute_dtype))
    LOGGER.warn('Variable dtype: {}'.format(policy.variable_dtype))


def init_tf_rlib(show=False, first=False):
    """
    first(bool): if True, some flags and in first time should be checked, such as log_path save_path. 
    """
    if FLAGS.log_path is None or not first:
        # rename log/save path
        FLAGS.log_path = os.path.join(FLAGS.local_path, FLAGS.exp_name,
                                      FLAGS.current_time, FLAGS.path_postfix,
                                      'log')
    if FLAGS.save_path is None or not first:
        FLAGS.save_path = os.path.join(FLAGS.local_path, FLAGS.exp_name,
                                       FLAGS.current_time, FLAGS.path_postfix,
                                       'ckpt')
    # logging config
    if FLAGS.purge_logs:
        purge_logs()

    if not os.path.exists(FLAGS.log_path):
        os.makedirs(FLAGS.log_path)

    hd = logging.logging.FileHandler(
        os.path.join(FLAGS.log_path, 'logging.txt'))
    hd.setLevel('INFO')
    LOGGER.addHandler(hd)
    tf.get_logger().setLevel('WARNING')
    LOGGER.setLevel(FLAGS.log_level)

    # envs
    if FLAGS.gpus is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpus
    # AMP
    set_amp(FLAGS.amp)
    # XLA
    set_xla(FLAGS.xla)

    if show:
        ## show all FLAGS
        for flag, value in FLAGS.flag_values_dict().items():
            LOGGER.info('FLAGS: --{}={}'.format(flag, value))
        if not FLAGS.xla:
            LOGGER.warn('enable --xla=True is recommended for ~10% speedup.')
