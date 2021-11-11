import numpy as np
import os
import random
import shutil
import os
import glob
import pandas as pd
import xml.etree.ElementTree as ET
from sklearn.model_selection import train_test_split

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
    globlist = glob.glob(path + '/*.xml')
    totalBoundingBoxes = 0
    idx = 0
    for xml_file in globlist:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        boundingBoxes = root.findall('object')
        totalBoundingBoxes += len(boundingBoxes)
    for xml_file in globlist:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        boundingBoxes = root.findall('object')

        if (len(boundingBoxes) == 0):
            # value = ("TRAINING",
            #          'image_set/' + root.find('filename').text,
            #          None,
            #          None,
            #          None,
            #          None,
            #          None,
            #          None,
            #          None,
            #          None,
            #          None,
            #          )
            # xml_list.append(value)
            pass
        else:
            for member in boundingBoxes:
                if idx > int(0.3 * totalBoundingBoxes):
                    set = "TRAINING"
                elif idx > int(0.1 * totalBoundingBoxes):
                    set = "VALIDATION"
                else:
                    set = "TEST"

                value = (set,
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
                idx += 1

    column_name = ['set', 'filename', 'class', 'xmin', 'ymin', None, None, 'xmax', 'ymax', None, None]
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    return xml_df

spec = model_spec.get('efficientdet_lite0')

print("started loading data set")

image_path = os.path.join(os.getcwd(), 'xml_dataset')
xml_df = xml_to_csv(image_path)
xml_df.to_csv('image_set.csv', index=None, header=False)
print('Successfully converted xml to csv.')
train_data, validation_data, test_data = object_detector.DataLoader.from_csv('./image_set.csv')


print("loaded data set...creating model")

model = object_detector.create(train_data, model_spec=spec, epochs=1, batch_size=1, train_whole_model=True, validation_data=validation_data)
print("created model.")

# model.evaluate(test_data)
print("exporting model")

model.export(export_dir='.')
print("evaluating model")

model.evaluate_tflite('model.tflite', test_data)

print("finished")
