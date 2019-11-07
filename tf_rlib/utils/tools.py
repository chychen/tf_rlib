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
    logging.warn('CUDA_VISIBLE_DEVICES={}'.format(FLAGS.gpus))
