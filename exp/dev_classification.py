"""
python dev_classification.py --depth=32 --model_alpha=48 --bottleneck=False
python dev_classification.py --depth=110 --model_alpha=48 --bottleneck=False --gpus=1 --port=6007
"""
import sys
sys.path.append('..')
import tf_rlib
from tf_rlib.runners import ClassificationRunner


def main():
    datasets = tf_rlib.datasets.get_cifar10()
    runner = ClassificationRunner(*datasets)
    runner.fit(FLGAS.epochs, FLAGS.lr)


if __name__ == '__main__':
    main()
