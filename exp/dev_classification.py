"""
python dev.py --depth=32 --model_alpha=48 --bottleneck=False
"""
import sys
sys.path.append('..')
import tf_rlib
from tf_rlib.runners import ClassificationRunner


def main():
    datasets = tf_rlib.datasets.get_cifar10()
    runner = ClassificationRunner(*datasets)
    runner.fit(300)


if __name__ == '__main__':
    main()
