import traceback
import os
import itertools
import tf_rlib
from multiprocessing import Pool, Queue, Value, Process
import tensorflow as tf
from tensorboard.plugins.hparams import api as hp
from absl import flags, logging

FLAGS = flags.FLAGS
LOGGER = logging.get_absl_logger()

KEY_NUM_GPUS = 'num_gpus'


class HParamTuner:
    """
    Example:
        hpt = HParamTuner(ClassificationRunner, ['0','1','2','3'], tf_rlib.datasets.get_cifar10)
        hpt(epochs=[100, 200, 300], lr=[1e-2, 1e-3, 1e-3])
    """
    def __init__(self, runner_cls, gpu_ids, dataset_fn):
        self.runner_cls = runner_cls
        self.gpu_ids = gpu_ids
        self.dataset_fn = dataset_fn
        self.session_counter = 0

    def __call__(self, trials=None, **kwargs):
        """
        Runs optimization across gpus with cuda drivers
        :param trials: List of Dicts like: [{lr:1e-1, wd:1e-4}, {lr:1e-2, wd:1e-4, bs:256}]
        :return:
        """
        if trials is None:
            trials = self.grid_serach(**kwargs)
        # append dataset_fn and path
        trials = [(x, self.dataset_fn, FLAGS.log_path) for x in trials]

        # build q of gpu ids so we can use them in each process
        # this is thread safe so each process can pull out a gpu id, run its task and put it back when done
        gpu_q = Queue()
        for gpu_id in self.gpu_ids:
            gpu_q.put(gpu_id)

        local_session_num = Value('i', self.session_counter, lock=True)

        # called by the Pool when a process starts
        def init(local_gpu_q, local_session_num):
            global g_gpu_id_q
            global session_num
            g_gpu_id_q = local_gpu_q
            session_num = local_session_num

        # init a pool with the nb of worker threads we want
        pool = Pool(processes=len(self.gpu_ids),
                    initializer=init,
                    initargs=(gpu_q, local_session_num))
        # apply parallelization
        results = pool.map(self._optimize_parallel_gpu, trials)
        self.session_counter = local_session_num.value
        pool.close()
        pool.join()
        return results

    def grid_serach(self, **kwargs):
        flat_params = []
        for k, values in kwargs.items():
            group = []
            for v in values:
                group.append({k: v})
            flat_params.append(group)

        combinations = list(itertools.product(*flat_params))
        trials = []
        for combination in combinations:
            temp = {}
            for one_dict in combination:
                for k, v in one_dict.items():
                    temp[k] = v
            trials.append(temp)

        return trials

    def train_function(self, trial_params, dataset_fn, log_path):
        for k, v in trial_params.items():
            if k in FLAGS.flag_values_dict() or k == KEY_NUM_GPUS:
                setattr(FLAGS, k, v)
            else:
                raise ValueError('--{} not in FLAGS'.format(k))

        with session_num.get_lock():
            session_num.value = session_num.value + 1
            FLAGS.path_postfix = 'hparam_{}'.format(session_num.value)

        datasets = dataset_fn()
        runner = self.runner_cls(*datasets)
        runner.fit(FLAGS.epochs, FLAGS.lr)
        with tf.summary.create_file_writer(FLAGS.log_path).as_default():
            # add log_path as a dummy feature avoid from reducing same parameters into one
            trial_params['log_path'] = FLAGS.log_path
            hp.hparams(trial_params)  # record the values used in this trial
            tf.summary.scalar(runner.best_state,
                              runner.best_state_record,
                              step=0)
        return

    def _optimize_parallel_gpu(self, args):
        trial_params, dataset_fn, log_path = args[0], args[1], args[2]
        if KEY_NUM_GPUS in trial_params:
            num_gpus = trial_params[KEY_NUM_GPUS]
        else:
            num_gpus = 1
        # get set of gpu ids
        gpu_id_set = []
        for _ in range(num_gpus):
            gpu_id_set.append(g_gpu_id_q.get(block=True))
        gpu_id_set = ",".join(gpu_id_set)
        try:
            # enable the proper gpus
            tf_rlib.utils.set_gpus(gpu_id_set)
            # run training fx on the specific gpus
            results = self.train_function(trial_params, dataset_fn, log_path)
            return [trial_params, results]
        except Exception as e:
            print('Caught exception in worker thread', e)
            # This prints the type, value, and stack trace of the
            # current exception being handled.
            traceback.print_exc()
            return [trial_params, None]
        finally:
            g_gpu_id_q.put(gpu_id_set)
