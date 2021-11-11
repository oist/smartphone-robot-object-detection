import numpy as np
import os
import random
import shutil
import os
import glob
import pandas as pd
import xml.etree.ElementTree as ET

from tflite_model_maker.config import ExportFormat
from tflite_model_maker import model_spec
from tflite_model_maker import object_detector

import tensorflow as tf
assert tf.__version__.startswith('2')

tf.get_logger().setLevel('ERROR')
from absl import logging
logging.set_verbosity(logging.ERROR)

labelDict = {'r':'puck_red','o':'puck_off', 'f':'robot_front', 'br':'robot_backright', 'bl':'robot_backleft'}

def xml_to_csv(path):
    xml_list = []
    for xml_file in glob.glob(path + '/*.xml'):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('object'):
            value = ("UNASSIGNED",
                'image_set/' + root.find('filename').text,
                labelDict.get(member[0].text),
                int(member[4][0].text),
                int(member[4][1].text),
                None,
                None,
                int(member[4][2].text),
                int(member[4][3].text),
                None,
                None,
                )
            xml_list.append(value)
    column_name = ['set', 'filename', 'class', 'xmin', 'ymin', None, None, 'xmax', 'ymax', None, None]
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    return xml_df

spec = model_spec.get('efficientdet_lite0')

print("started loading data set")

# split data into training and testing set
# if os.path.exists('./train/images'):
#   shutil.rmtree('./train/images')
# if os.path.exists('./train/annotations'):
#   shutil.rmtree('./train/annotations')
# if os.path.exists('./test/images'):
#   shutil.rmtree('./test/images')
# if os.path.exists('./test/annotations'):
#   shutil.rmtree('./test/annotations')
#
# os.makedirs('./train/images')
# os.makedirs('./train/annotations')
# os.makedirs('./test/images')
# os.makedirs('./test/annotations')
#
# image_paths = os.listdir('./image_set')
# random.shuffle(image_paths)
#
# for i, image_path in enumerate(image_paths):
#   if i < int(len(image_paths) * 0.8):
#     if os.path.exists(f'./xml_dataset/{image_path.replace("jpg", "xml")}'): # Not all images have objects in them
#       shutil.copy(f'./image_set/{image_path}', './train/images')
#       shutil.copy(f'./xml_dataset/{image_path.replace("jpg", "xml")}', './train/annotations')
#   else:
#     if os.path.exists(f'./xml_dataset/{image_path.replace("jpg", "xml")}'):
#       shutil.copy(f'./image_set/{image_path}', './test/images')
#       shutil.copy(f'./xml_dataset/{image_path.replace("jpg", "xml")}', './test/annotations')
#
# train_data = object_detector.DataLoader.from_pascal_voc('./train/images', './train/annotations', ['r', 'o', 'f', 'br','bl'])
# test_data = object_detector.DataLoader.from_pascal_voc('./test/images', './test/annotations', ['r', 'o', 'f', 'br','bl'])

image_path = os.path.join(os.getcwd(), 'xml_dataset')
xml_df = xml_to_csv(image_path)
xml_df.to_csv('image_set.csv', index=None, header=False)
print('Successfully converted xml to csv.')
train_data, validation_data, test_data = object_detector.DataLoader.from_csv('./image_set.csv')


print("loaded data set...creating model")

model = object_detector.create(train_data, model_spec=spec, epochs=1, batch_size=1, train_whole_model=True, validation_data=validation_data)
print("created model...evaluating model")

model.evaluate(test_data)
print("exporting model")

model.export(export_dir='.')
print("evaluating model")

model.evaluate_tflite('model.tflite', test_data)

print("finished")
