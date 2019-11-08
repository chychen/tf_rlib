"""
python benchmark.py --benchmark_runner=ClassificationResNet18Cifar10
"""
import sys
sys.path.append('..')
import tf_rlib
from tf_rlib.runners.benchmark import ClassificationResNet18Cifar10, ClassificationPyramidNet272Cifar10
from absl import flags

FLAGS = flags.FLAGS


def main():
    if FLAGS.benchmark_runner == 'ClassificationResNet18Cifar10':
        runner = ClassificationResNet18Cifar10()
    elif FLAGS.benchmark_runner == 'ClassificationPyramidNet272Cifar10':
        runner = ClassificationPyramidNet272Cifar10()
    else:
        raise NotImplementedError


if __name__ == '__main__':
    main()
