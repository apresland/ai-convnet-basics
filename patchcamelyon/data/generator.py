
from base.base_data_generator import BaseDataGenerator
import os
import numpy as np
import pandas as pd
import random

from glob import glob
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator

class DataGenerator(BaseDataGenerator):
    def __init__(self, config):

        self.IMG_SIZE = 96
        self.BATCH_SIZE = 16

        path = "/data/patchcamelyon/" 
        labels = pd.read_csv(path + 'train_labels.csv')
        train_path = path + 'train/'
        test_path = path + 'test/'

        df = pd.DataFrame({'path': glob(os.path.join(train_path,'*.tif'))})
        df['id'] = df.path.map(lambda x: ((x.split("n")[2].split('.')[0])[1:]))
        df = df.merge(labels, on = "id")
        df = df[df['id'] != 'dd6dfed324f9fcb6f93f46f32fc800f2ec196be2']
        df = df[df['id'] != '9369c7278ec8bcc6c880d99194de09fc2bd4efbe']
        df = df[df['id'] != '9369c7278ec8bcc6c880d99194de09fc2bd4efbe']
        df = df[df['id'] != 'f6f1d771d14f7129a6c3ac2c220d90992c30c10b']
        df = df[df['id'] != '9071b424ec2e84deeb59b54d2450a6d0172cf701']
        df = df[df['id'] != 'c448cd6574108cf14514ad5bc27c0b2c97fc1a83']
        df = df[df['id'] != '54df3640d17119486e5c5f98019d2a92736feabc']
        df = df[df['id'] != '5f30d325d895d873d3e72a82ffc0101c45cba4a8']
        df = df[df['id'] != '5a268c0241b8510465cb002c4452d63fec71028a']
        df['label'] = df['label'].astype(str)

        train, test = train_test_split(df, test_size=0.2, stratify = df['label'])
        test, valid = train_test_split(test, test_size=0.5, stratify = test['label'])

        train_image_generator = ImageDataGenerator(
            rescale=1./255,
            vertical_flip = True,
            horizontal_flip = True,
            rotation_range=5,
            zoom_range=0.1,
            #brightness_range=(0.8, 1.2),
            #channel_shift_range=16,
            width_shift_range=0.1,
            height_shift_range=0.1)

        test_image_generator = ImageDataGenerator(
            rescale = 1./255) 

        self.train_data_generator = train_image_generator.flow_from_dataframe(
            dataframe = train, 
            directory = None,
            x_col = 'path', 
            y_col = 'label',
            target_size = (self.IMG_SIZE, self.IMG_SIZE),
            class_mode = "binary",
            batch_size=self.BATCH_SIZE,
            seed = 110318,
            shuffle = True)

        self.valid_data_generator = train_image_generator.flow_from_dataframe(
            dataframe = valid,
            directory = None,
            x_col = 'path',
            y_col = 'label',
            target_size = (self.IMG_SIZE, self.IMG_SIZE),
            class_mode = 'binary',
            batch_size = self.BATCH_SIZE,
            shuffle = True)

        self.test_data_generator = test_image_generator.flow_from_dataframe(
            dataframe = test,
            directory = None,
            x_col = 'path',
            y_col = 'label',
            target_size = (self.IMG_SIZE, self.IMG_SIZE),
            class_mode = 'binary',
            batch_size = 1,
            shuffle = False)

