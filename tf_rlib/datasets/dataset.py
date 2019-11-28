import tensorflow as tf
import numpy as np
import os
import glob
import pandas as pd
from tqdm.auto import tqdm
from absl import flags, logging
from PIL import Image

FLAGS = flags.FLAGS
LOGGER = logging.get_absl_logger()


class Dataset:
    def __init__(self):
        pass

    def get_data(self):
        raise NotImplementedError

    def get_df(self):
        raise NotImplementedError
