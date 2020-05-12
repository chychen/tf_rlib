from absl import flags
from absl import logging

FLAGS = flags.FLAGS
LOGGER = logging.get_absl_logger()


class ESRoPLRScheduler:
    """ Early Stop Reduce On Plateau LR Scheduler
    usage example:
    
    def init(self):
        self.lr_schler = ESRoPLRScheduler(self, FLAGS.lr)
        ...

    def begin_epoch_callback(self, epoch_id, epochs):
        if epoch_id < FLAGS.warmup:
            self.optim.learning_rate = (epoch_id+1) / FLAGS.warmup * self.init_lr
        else:
            self.optim.learning_rate = self.lr_schler.get_lr()
        ...
    """
    def __init__(self, runner_ctx, init_lr):
        """
        runner_ctx: runner class
        factor: factor by which the learning rate will be reduced. new_lr = lr * factor
        patience: number of epochs with no improvement after which learning rate will be reduced.
        """
        self.runner_ctx = runner_ctx
        self.factor = FLAGS.lr_factor
        self.patient = FLAGS.lr_patience
        self.patient_counter = 0
        self.lr = init_lr

    def get_lr(self):
        if self.runner_ctx.metrics_manager.is_better_state():
            self.patient_counter = 0
        else:
            self.patient_counter = self.patient_counter + 1
            if self.patient_counter >= self.patient:
                self.patient_counter = 0
                self.lr = self.lr * self.factor

        if self.optim.learning_rate <= self.init_lr / 1e3:
            LOGGER.warn('Early Stop')
            raise Exception
        return self.lr
