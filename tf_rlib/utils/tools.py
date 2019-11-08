import os
import shutil
import tensorflow as tf
from absl import flags, logging

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
    FLAGS.log_path = os.path.join(FLAGS.local_path, FLAGS.exp_name,
                                  FLAGS.current_time, FLAGS.log_path)
    if not os.path.exists(FLAGS.log_path):
        os.makedirs(FLAGS.log_path)
    LOGGER.warn('update FLAGS: --{}={}'.format('log_path', FLAGS.log_path))
    FLAGS.save_path = os.path.join(FLAGS.local_path, FLAGS.exp_name,
                                   FLAGS.current_time, FLAGS.save_path)
    LOGGER.warn('update FLAGS: --{}={}'.format('save_path', FLAGS.save_path))


def set_exp_name(name):
    FLAGS.exp_name = name
    reset_paths()


def set_xla(enable):
    FLAGS.xla = enable
    tf.keras.backend.clear_session()
    tf.config.optimizer.set_jit(FLAGS.xla)


def init_tf_rlib(show=False):
    # rename log/save path
    FLAGS.log_path = os.path.join(FLAGS.local_path, FLAGS.exp_name,
                                  FLAGS.current_time, FLAGS.log_path)
    FLAGS.save_path = os.path.join(FLAGS.local_path, FLAGS.exp_name,
                                   FLAGS.current_time, FLAGS.save_path)
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

    ## show all FLAGS
    if show:
        for flag, value in FLAGS.flag_values_dict().items():
            LOGGER.info('FLAGS: --{}={}'.format(flag, value))

    # envs
    os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpus
    # enable XLA
    tf.keras.backend.clear_session()
    tf.config.optimizer.set_jit(FLAGS.xla)
