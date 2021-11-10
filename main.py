import numpy as np
import os
import random
import shutil

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
# train_data, validation_data, test_data = object_detector.DataLoader.from_csv('./salads_ml_use.csv')

# split data into training and testing set
if os.path.exists('./train/images'):
  os.removedirs('./train/images')
if os.path.exists('./train/annotations'):
  os.removedirs('./train/annotations')
if os.path.exists('./test/images'):
  os.removedirs('./test/images')
if os.path.exists('./test/annotations'):
  os.removedirs('./test/annotations')

os.makedirs('./train/images')
os.makedirs('./train/annotations')
os.makedirs('./test/images')
os.makedirs('./test/annotations')

image_paths = os.listdir('./image_set')
random.shuffle(image_paths)

for i, image_path in enumerate(image_paths):
  if i < int(len(image_paths) * 0.8):
    if os.path.exists(f'./xml_dataset/{image_path.replace("jpg", "xml")}'): # Not all images have objects in them
      shutil.copy(f'./image_set/{image_path}', './train/images')
      shutil.copy(f'./xml_dataset/{image_path.replace("jpg", "xml")}', './train/annotations')
  else:
    if os.path.exists(f'./xml_dataset/{image_path.replace("jpg", "xml")}'):
      shutil.copy(f'./image_set/{image_path}', './test/images')
      shutil.copy(f'./xml_dataset/{image_path.replace("JPG", "xml")}', './test/annotations')

train_data = object_detector.DataLoader.from_pascal_voc('./train/images', './train/annotations', ['r', 'o', 'f', 'br','bl'])
test_data = object_detector.DataLoader.from_pascal_voc('./test/images', './test/annotations', ['r', 'o', 'f', 'br','bl'])

print("loaded data set...creating model")

model = object_detector.create(train_data, model_spec=spec, epochs=8, batch_size=8, train_whole_model=True, validation_data=test_data)
print("created model...evaluating model")

model.evaluate(test_data)
print("exporting model")

model.export(export_dir='.')
print("evaluating model")

model.evaluate_tflite('model.tflite', test_data)

print("finished")
