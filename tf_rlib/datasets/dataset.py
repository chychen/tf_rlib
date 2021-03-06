from absl import flags, logging

FLAGS = flags.FLAGS
LOGGER = logging.get_absl_logger()


class Dataset:
    def __init__(self):
        pass

    def get_data(self):
        raise NotImplementedError

    def get_df(self):
        raise NotImplementedError

    def vis(self, num_samples):
        raise NotImplementedError
