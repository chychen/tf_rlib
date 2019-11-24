"""
python dev_adae_gridsearch.py
"""

import sys
sys.path.append('..')
import tf_rlib
from tf_rlib.runners import ADAERunner
from tf_rlib.utils import HParamTuner

tf_rlib.utils.purge_logs()

FLAGS = tf_rlib.FLAGS
FLAGS.bs = 8
FLAGS.out_dim = 3
FLAGS.log_level = 'WARN'

mvtec = tf_rlib.datasets.MVTecDS()


def get_data():
    return mvtec.get_data('toothbrush', target_size=(128, 128))


hpt = HParamTuner(ADAERunner, ['0', '1', '2', '3'], get_data)
hpt(bs=[4, 8, 16],
    latent_dim=[10, 50, 100, 200],
    epochs=[300, 600, 1200],
    loss_fn=[key for key in ADAERunner.LOSSES_POOL])
