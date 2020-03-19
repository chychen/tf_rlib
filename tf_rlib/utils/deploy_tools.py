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
    
def convert_to_TFTRT(saved_model_path, target_path,
                     max_workspace_size_bytes=1<<22,
                     precision='FP16',
                     minimum_segment_size=3,
                     is_dynamic_op=True,
                     use_calibration=True,
                     max_batch_size=32,
                     calibration_input_fn=None):
    '''
    Args:
        precision: 'FP32', 'FP16' or 'INT8'
        calibration_input_fn: INT8 calibration is needed. A generator function that yields input data as a
            list or tuple, which will be used to execute the converted signature for
            calibration. All the returned input data should have the same shape.
    Source: https://docs.nvidia.com/deeplearning/frameworks/tf-trt-user-guide/index.html
    '''
    conversion_params = trt.TrtConversionParams(rewriter_config_template=None,
                                                maximum_cached_engines=1,
                                                max_workspace_size_bytes=max_workspace_size_bytes,
                                                precision_mode=precision,
                                                minimum_segment_size=minimum_segment_size,
                                                is_dynamic_op=is_dynamic_op,
                                                use_calibration=use_calibration,
                                                max_batch_size=max_batch_size)
    converter = trt.TrtGraphConverterV2(input_saved_model_dir=saved_model_path, conversion_params=conversion_params)
    converter.convert()
    converter.save(target_path)