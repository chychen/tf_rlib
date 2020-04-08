"""
python benchmark.py --benchmark=Classification.ResNet18Cifar10
"""
import sys
sys.path.append('..')
import tf_rlib
from tf_rlib.runners.benchmark import Classification
from tf_rlib.runners.benchmark import FewShot
from tf_rlib.runners.benchmark import OutOfDistribution
from absl import flags

FLAGS = flags.FLAGS


def main():
    # Classification
    if FLAGS.benchmark is None or FLAGS.benchmark == 'Classification.PyramidNet10Cifar10':
        runner = Classification.PyramidNet10Cifar10()
    elif FLAGS.benchmark == 'Classification.PyramidNet272Cifar10':
        runner = Classification.PyramidNet272Cifar10()
    elif FLAGS.benchmark == 'Classification.ResNet18Cifar10':
        runner = Classification.ResNet18Cifar10()
    elif FLAGS.benchmark == 'Classification.ResNet18PreactCifar10':
        runner = Classification.ResNet18PreactCifar10()
    elif FLAGS.benchmark == 'Classification.ResNet18PreactLastnormCifar10':
        runner = Classification.ResNet18PreactLastnormCifar10()
    # FewShot
    elif FLAGS.benchmark == 'FewShot.RelationNetOmniglot':
        runner = FewShot.RelationNetOmniglot()
    # OutOfDistribution
    elif FLAGS.benchmark == 'OutOfDistribution.SvdRndCifar10vsSVHN':
        runner = OutOfDistribution.SvdRndCifar10vsSVHN()
    else:
        raise NotImplementedError


if __name__ == '__main__':
    main()
