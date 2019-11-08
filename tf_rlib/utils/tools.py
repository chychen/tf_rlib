import os
import shutil
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
