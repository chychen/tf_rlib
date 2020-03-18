import tf_rlib
import tensorflow as tf
import numpy as np
import os
from tensorflow.python.compiler.tensorrt import trt_convert as trt

FLAGS = tf_rlib.FLAGS

def save_weights(runner, path):
    runner.model.save_weights(path)

def load_weights(runner, path):
    runner.model.load_weights(path)
    
def save_as_SavedModel(runner, path):
    if FLAGS.amp:
        save_weights(runner, './temp.h5')
        tf_rlib.utils.set_amp(False)
        # initialize the model with float32
        runner.init()
        if len(runner.model_inputs)==1:
            runner.model(runner.model_inputs[0])
        else:
            runner.model(runner.model_inputs)
        load_weights(runner, './temp.h5')
        os.system('rm ./temp.h5')
    runner.model.save(path, save_format='tf')
    
def convert_to_TFTRT(saved_model_path, target_path):
    '''
    Source: https://docs.nvidia.com/deeplearning/frameworks/tf-trt-user-guide/index.html
    '''
    converter = trt.TrtGraphConverterV2(input_saved_model_dir=saved_model_path)
    converter.convert()
    converter.save(target_path)