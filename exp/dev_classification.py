"""
python dev_classification.py --depth=32 --model_alpha=48 --bottleneck=False
python dev_classification.py --depth=110 --model_alpha=48 --bottleneck=False --gpus="2"
python dev_classification.py --depth=272 --model_alpha=200 --bottleneck=True --gpus="0,1,2,3" --bs=128 --xla=True --log_level="INFO" --local_path="/ws_data/bkp/pynet272/" --lr=1e-1 --wd=1e-4 --l2=0.0 --exp_name="SGDW"
"""
import sys
sys.path.append('..')
import tf_rlib
from tf_rlib.runners import ClassificationRunner
from absl import flags

FLAGS = flags.FLAGS


def main():
    datasets = tf_rlib.datasets.get_cifar10()
    runner = ClassificationRunner(*datasets)
    runner.fit(FLAGS.epochs, FLAGS.lr)


if __name__ == '__main__':
    main()
