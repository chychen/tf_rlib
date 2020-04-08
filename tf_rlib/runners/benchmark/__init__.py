# Classification
from tf_rlib.runners.benchmark.classification_resnet18 import ClassificationResNet18Cifar10, ClassificationResNet18PreactCifar10, ClassificationResNet18PreactLastnormCifar10
from tf_rlib.runners.benchmark.classification_pyramidnet272 import ClassificationPyramidNet272Cifar10
from tf_rlib.runners.benchmark.classification_pyramidnet10 import ClassificationPyramidNet10Cifar10
# FewShot
from tf_rlib.runners.benchmark.fewshot_relationnet import FewShotRelationNetOmniglot
# OutOfDistribution
from tf_rlib.runners.benchmark.ood_svd_rnd import OODSvdRndCifar10vsSVHN


class Classification:
    ResNet18Cifar10 = ClassificationResNet18Cifar10
    ResNet18PreactCifar10 = ClassificationResNet18PreactCifar10
    ResNet18PreactLastnormCifar10 = ClassificationResNet18PreactLastnormCifar10
    PyramidNet272Cifar10 = ClassificationPyramidNet272Cifar10
    PyramidNet10Cifar10 = ClassificationPyramidNet10Cifar10


class FewShot:
    RelationNetOmniglot = FewShotRelationNetOmniglot


class OutOfDistribution:
    SvdRndCifar10vsSVHN = OODSvdRndCifar10vsSVHN
