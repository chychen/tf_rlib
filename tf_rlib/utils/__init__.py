from tf_rlib.utils.hparam_tuner import HParamTuner
from tf_rlib.utils.tools import init_tf_rlib, purge_logs, set_logging, set_gpus, set_exp_name, set_amp
from tf_rlib.utils.onplateau_lrscheduler import ESRoPLRScheduler


class dummy_context_mgr:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc_value, traceback):
        return False
