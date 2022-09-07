import tensorflow as tf

import matplotlib.pyplot as plt

import collections
import random
import numpy as np
import os 
import time
import json
from PIL import Image

import argparse

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-folder', default='./data/')
    args = parser.parse_args()
    return args


args = parse()

annotation_folder = args.data_folder + './annotation'

if not os.path.exists(annotation_folder):
    annotation_zip = tf.keras.utils.get_file('captions.zip',
                                             cache_subdir=os.path.abspath(args.data_folder),
                                             origin='http://images.cocodataset.org/annotations/annotations_trainval2014.zip',
                                             extract=True,
                                            )
    os.remove(annotation_zip)


image_folder = args.data_folder + './image'
if not os.path.exists(image_folder):
    image_zip = tf.keras.utils.get_file('train2014.zip',
                                        cache_subdir=os.path.abspath(args.data_folder),
                                        origin='http://images.cocodataset.org/zips/train2014.zip',
                                        extract=True)
    image_path = os.path.dirname(image_zip) + image_folder
    os.remove(image_zip)
else:
    image_path = args.data_folder

