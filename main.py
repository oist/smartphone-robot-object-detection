import numpy as np
import os

from tflite_model_maker.config import ExportFormat
from tflite_model_maker import model_spec
from tflite_model_maker import object_detector

import tensorflow as tf
assert tf.__version__.startswith('2')

tf.get_logger().setLevel('ERROR')
from absl import logging
logging.set_verbosity(logging.ERROR)

spec = model_spec.get('efficientdet_lite0')

print("started loading data set")
train_data, validation_data, test_data = object_detector.DataLoader.from_csv('./salads_ml_use.csv')
train_data, validation_data, test_data = object_detector.DataLoader.

print("loaded data set...creating model")

model = object_detector.create(train_data, model_spec=spec, batch_size=8, train_whole_model=True, validation_data=validation_data)
print("created model...evaluating model")

model.evaluate(test_data)
print("exporting model")

model.export(export_dir='.')
print("evaluating model")

model.evaluate_tflite('model.tflite', test_data)

print("finished")
