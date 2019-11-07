"""
python dev_classification_gridsearch.py
"""
import sys
sys.path.append('..')
import tf_rlib
from tf_rlib.runners import ClassificationRunner
from tf_rlib.utils import HParamTuner

FLAGS = tf_rlib.FLAGS


def main():
    FLAGS.depth = 32
    FLAGS.model_alpha = 48
    FLAGS.bottleneck = False
    tf_rlib.utils.set_logging('INFO')
    hpt = HParamTuner(ClassificationRunner, ['0', '1', '2', '3'],
                      tf_rlib.datasets.get_cifar10)
    hpt(epochs=[
        5,
    ],
        lr=[1e-2, 1e-3, 1e-4],
        bs=[128, 256],
        l2=[1e-4, 1e-3],
        warmup=[
            0,
        ])


if __name__ == '__main__':
    main()
