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
    hpt = HParamTuner(ClassificationRunner, ['0', '1'],
                      tf_rlib.datasets.get_cifar10)
    hpt(epochs=[300], xla=[True, False, True, False, True, False])


#     tf_rlib.utils.set_logging('DEBUG')
#     FLAGS.lr=1e-3
#     FLAGS.bs=128
#     # https://github.com/dyhan0920/PyramidNet-PyTorch
#     FLAGS.depth=272
#     FLAGS.model_alpha=200
#     FLAGS.bottleneck=True
#     hpt = HParamTuner(ClassificationRunner, ['0,1,2,3'],
#                       tf_rlib.datasets.get_cifar10)
#     hpt(xla=[True, False, True, False])

if __name__ == '__main__':
    main()
