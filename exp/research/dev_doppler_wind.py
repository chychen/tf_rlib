import sys
sys.path.append('../..')
import tf_rlib
from tf_rlib.runners.research import RegHResResNet18Runner
FLAGS = tf_rlib.FLAGS

tf_rlib.utils.purge_logs()
tf_rlib.utils.set_gpus('0, 1, 2, 3')
tf_rlib.utils.set_logging('WARN')
tf_rlib.utils.set_amp(False)
amp_factor = 2 if FLAGS.amp else 1

FLAGS.loss_fn = 'mse'
FLAGS.bs = 128 * amp_factor * FLAGS.num_gpus
FLAGS.dim = 2
FLAGS.out_dim = 4
FLAGS.lr = 1e-3 * amp_factor * FLAGS.num_gpus
FLAGS.steps_per_epoch = 1000 // amp_factor // FLAGS.num_gpus
FLAGS.epochs = 300

DopplerWind = tf_rlib.datasets.DopplerWind()
train_dset, valid_dset = DopplerWind.get_data()
runner = RegHResResNet18Runner(train_dset,
                               valid_dset,
                               y_denorm_fn=DopplerWind.get_y_denorm_fn())
runner.fit(FLAGS.epochs, lr=FLAGS.lr)
